# -*- coding: utf-8 -*-
"""
Policy-gradient trainer for the Adaptive Selector (REINFORCE + critic).

Per training step (batch of videos):
    1. pruning.forward: lightweight features -> selector (SELECT/STOP) ->
       heavy CLIP on the SELECTED frames only -> features   [selector has grads]
    2. captions = frozen captioner, greedy decode           [no_grad]
    3. rewards r_t (CaptionReward, paper formula; CIDEr never backprops)
    4. G_t = discounted returns (gamma = 1.0)
    5. V_hat_t = critic(h_t);  A_t = G_t - stop_gradient(V_hat_t)
    6. L_policy = -E[ sum_t A_t log pi(a_t | h_t) ]
       L_util   = 1/2 * mean (V_hat_t - G_t)^2
       L = L_policy + lambda_utility * L_util

The captioner (and the heavy CLIP encoder) are FROZEN: only the selector,
the lightweight encoder, and the critic train. The captioner may later be
loaded from a real captioning checkpoint (config: captioner.checkpoint).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from models.frame_cocap.frame_video_captioner import generate_caption
from models.selectors.adaptive_frame_pruning import AdaptiveFramePruning
from models.selectors.caption_reward import CaptionReward, discounted_returns
from models.selectors.utility_critic import UtilityCritic

logger = logging.getLogger(__name__)

__all__ = ["SelectorPolicyTrainer", "load_captioner_checkpoint"]


def load_captioner_checkpoint(captioner: nn.Module, path: str, device: Optional[torch.device] = None):
    """Load a (future) captioning checkpoint into the Frame-CoCap captioner.

    Accepts {"captioner": state_dict}, {"model": state_dict}, or a bare
    state dict. The captioner (CLIP-initialized CaptionHead) currently has
    NO trained checkpoint in this project, so selector training must wait
    for a real captioning checkpoint to produce meaningful CIDEr.
    """
    state = torch.load(path, map_location=device or "cpu")
    if isinstance(state, dict) and "captioner" in state:
        sd = state["captioner"]
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state
    missing, unexpected = captioner.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("captioner checkpoint missing keys (%d): %s ...",
                       len(missing), str(missing[:5]))
    if unexpected:
        logger.warning("captioner checkpoint unexpected keys (%d): %s ...",
                       len(unexpected), str(unexpected[:5]))
    return state


class SelectorPolicyTrainer:
    def __init__(
            self,
            pruning: AdaptiveFramePruning,
            captioner: nn.Module,
            critic: UtilityCritic,
            reward: CaptionReward,
            lr_selector: float = 1e-4,
            lr_critic: float = 1e-4,
            lambda_utility: float = 1.0,
            gamma: float = 1.0,
            lr_decay_gamma: float = 0.95,
            device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pruning = pruning
        self.captioner = captioner
        self.critic = critic
        self.reward = reward
        self.gamma = gamma
        self.lambda_utility = lambda_utility
        # method markers (asserted by the training script's preflight)
        self.step_wise_reward = True          # r_t = dQ_t - lambda_c at EVERY select step
        self.advantage_normalization = True   # A_norm over valid steps for the policy loss

        # captioner + heavy encoder frozen (heavy encoder is inside pruning,
        # already under no_grad in forward; captioner is explicitly frozen)
        self.captioner.eval()
        for p in self.captioner.parameters():
            p.requires_grad_(False)

        selector_params = (list(pruning.selector.parameters())
                           + list(pruning.lightweight_encoder.parameters()))
        self.opt_selector = torch.optim.Adam(selector_params, lr=lr_selector)
        self.opt_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)
        # epoch-based multiplicative decay matched to the formal run length
        # (official CoCap design principle: lr(e) = lr0 * gamma^e; the
        # diagnostic's StepLR(5, 0.5) decayed too fast for 20 epochs)
        self.lr_decay_gamma = lr_decay_gamma
        self.scheduler_selector = torch.optim.lr_scheduler.LambdaLR(
            self.opt_selector, lr_lambda=lambda e: lr_decay_gamma ** e)
        self.scheduler_critic = torch.optim.lr_scheduler.LambdaLR(
            self.opt_critic, lr_lambda=lambda e: lr_decay_gamma ** e)

    # ------------------------------------------------------------------ step
    def train_step(self, candidate_frames: Tensor, refs: Sequence[Sequence[str]]) -> Dict[str, float]:
        """One REINFORCE step. candidate_frames: (B, M, 3, H, W)."""
        self.opt_selector.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True)

        out = self.pruning(candidate_frames)          # selector path has grads
        counts = out["selected_count"]
        # health invariant: the heavy CLIP encoder must have seen exactly
        # the selected frames (never the full candidate pool)
        assert int(out["heavy_frames_encoded"]) == int(counts.sum()), \
            f"heavy CLIP encoded {out['heavy_frames_encoded']} frames != " \
            f"{int(counts.sum())} selected"
        valid = out["selected_mask"].float()          # (B, S) selection steps
        B = candidate_frames.shape[0]
        S = self.reward.max_selected_frames

        # ---- step-wise credit (paper "full" formula, unchanged):
        #      every SELECT step t gets r_t = dQ_t - lambda_c,
        #      Q_t = CIDEr(caption from the first t+1 selected frames),
        #      Q_0 = 0, r_STOP = 0  ->  R = sum r_t = Q_N - lambda_c * N ----
        sel_frames = out["selected_frames"]               # (B, N_max, 3, H, W)
        sel_frame_mask = out["selected_frame_mask"]       # (B, N_max) long
        C, H, W = sel_frames.shape[2:]
        rewards = torch.zeros(B, S, device=counts.device)
        q_prev = torch.zeros(B, device=counts.device)
        max_count = int(counts.max())
        t0 = time.time()
        for t in range(max_count):
            active = counts > t
            if not active.any():
                break
            # prefix = first t+1 selected frames; heavy CLIP encodes ONLY the
            # valid prefix rows (no_grad, chunked), same as the main path
            prefix_frames = sel_frames[:, :t + 1].contiguous()       # (B, t+1, C, H, W)
            prefix_mask = sel_frame_mask[:, :t + 1]                  # (B, t+1) long
            flat = prefix_frames.reshape(B * (t + 1), C, H, W)
            valid_rows = prefix_mask.reshape(-1).bool()
            with torch.no_grad():
                heavy = self.pruning._encode_heavy(flat[valid_rows])
            pf = torch.zeros(B * (t + 1), heavy.shape[-1],
                             dtype=heavy.dtype, device=heavy.device)
            pf[valid_rows] = heavy
            pf = (pf.reshape(B, t + 1, -1) * prefix_mask.unsqueeze(-1))
            # caption of the prefix (batched, frozen captioner, no grad)
            caps_t = generate_caption(self.captioner, visual_features=pf,
                                      frame_mask=prefix_mask)
            q_t = torch.zeros(B, device=counts.device)
            for b in range(B):
                if active[b]:
                    q_t[b] = self.reward.cider_scores([caps_t[b]], [refs[b]])[0]
            rewards[active, t] = q_t[active] - q_prev[active] - self.reward.lambda_cost
            q_prev[active] = q_t[active]
        reward_time = time.time() - t0

        # returns + critic + losses
        G = discounted_returns(rewards, gamma=self.gamma)
        V = self.critic(out["hidden_states"]).squeeze(-1)   # (B, S)
        loss_util = self.critic.utility_loss(V.unsqueeze(-1), G, valid)
        # ---- advantage normalization (variance reduction): only VALID
        #      SELECT steps participate in mean/std; critic target is the
        #      raw return (unchanged) ----
        adv_raw = (G - V.detach()) * valid
        n_valid = max(int(valid.sum()), 1)
        adv_mean = adv_raw.sum() / n_valid
        adv_std = torch.sqrt(((adv_raw - adv_mean * valid) ** 2 * valid).sum() / n_valid)
        A_norm = (adv_raw - adv_mean * valid) / (adv_std + 1e-8)
        loss_policy = -(A_norm * out["action_log_probs"]).sum() / B
        loss = loss_policy + self.lambda_utility * loss_util

        loss.backward()
        # gradient norm over ALL trained params (selector + lightweight + critic),
        # measured only — no clipping (method unchanged)
        grad_norm = 0.0
        for p in list(self.pruning.selector.parameters()) + \
                list(self.pruning.lightweight_encoder.parameters()) + \
                list(self.critic.parameters()):
            if p.grad is not None:
                grad_norm += float((p.grad.detach() ** 2).sum())
        grad_norm = grad_norm ** 0.5
        self.opt_selector.step()
        self.opt_critic.step()

        n_valid = max(int(valid.sum()), 1)
        dist = {}
        for c in range(2, 17):
            n_c = int((counts == c).sum())
            if n_c:
                dist[str(c)] = n_c
        fracs = {
            "end_at_2": float((counts == 2).float().mean().detach()),
            "end_at_3_4": float(((counts >= 3) & (counts <= 4)).float().mean().detach()),
            "end_at_5_8": float(((counts >= 5) & (counts <= 8)).float().mean().detach()),
            "end_at_9_16": float(((counts >= 9) & (counts <= 16)).float().mean().detach()),
        }
        return {
            "policy_loss": float(loss_policy.detach()),
            "critic_loss": float(loss_util.detach()),
            "total_loss": float(loss.detach()),
            "mean_reward": float(rewards.sum(1).mean().detach()),
            "mean_return": float(G[:, 0].mean().detach()),
            "mean_advantage_raw": float(adv_raw.abs().sum().detach() / n_valid),
            "mean_advantage_normalized": float(A_norm.abs().sum().detach() / n_valid),
            "advantage_std": float(adv_std.detach()),
            "positive_reward_ratio": float(((rewards > 0).float() * valid).sum().detach() / n_valid),
            "mean_cider": float((rewards.sum(1) + self.reward.lambda_cost * counts).mean().detach()),
            "mean_selected_frames": float(counts.float().mean().detach()),
            "selection_ratio": float((counts.float() / candidate_frames.shape[1]).mean().detach()),
            "selected_count_distribution": dist,
            "episode_length_fractions": fracs,
            "mean_stop_step": float(out["stop_step"].float().mean().detach()),
            "min_selected": int(counts.min()),
            "max_selected": int(counts.max()),
            "patterns": [",".join(str(int(i)) for i in
                                  out["selected_indices"][b, out["selected_mask"][b]].tolist())
                         for b in range(B)],  # v2 logging: per-sample selected index patterns
            "heavy_frames_encoded": int(out["heavy_frames_encoded"]),
            "grad_norm": grad_norm,
            "lr_selector": self.opt_selector.param_groups[0]["lr"],
            "lr_critic": self.opt_critic.param_groups[0]["lr"],
            "reward_time_s": reward_time,
        }

    def scheduler_step(self):
        self.scheduler_selector.step()
        self.scheduler_critic.step()

    # ------------------------------------------------------------- checkpoint
    def save_checkpoint(self, path: str, epoch: int, config: Dict[str, Any], extra: Optional[Dict] = None):
        state = {
            "selector": self.pruning.selector.state_dict(),
            "lightweight_encoder": self.pruning.lightweight_encoder.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer_selector": self.opt_selector.state_dict(),
            "optimizer_critic": self.opt_critic.state_dict(),
            "scheduler_selector": self.scheduler_selector.state_dict(),
            "scheduler_critic": self.scheduler_critic.state_dict(),
            "epoch": epoch,
            "config": config,
        }
        if extra:
            state["stats"] = extra
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.pruning.selector.load_state_dict(state["selector"])
        self.pruning.lightweight_encoder.load_state_dict(state["lightweight_encoder"])
        self.critic.load_state_dict(state["critic"])
        if "optimizer_selector" in state:
            self.opt_selector.load_state_dict(state["optimizer_selector"])
            self.opt_critic.load_state_dict(state["optimizer_critic"])
            self.scheduler_selector.load_state_dict(state["scheduler_selector"])
            self.scheduler_critic.load_state_dict(state["scheduler_critic"])
        return state
