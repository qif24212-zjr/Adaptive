"""PickNet-style sequential Pick/Drop frame selector.

NOT an official PickNet reproduction: we use the pre-extracted 768-d frame
features of our captioning pipeline instead of PickNet's original low-resolution
grayscale glance input, and v1 trains through the captioning XE loss with a
straight-through estimator rather than PickNet's REINFORCE protocol.

Mechanism (sequential scan, variable-length):
    h_t   = GRUCell(x_t, h_{t-1})                     # sequential state
    p_t   = softmax(MLP([h_t ; x_t]))[PICK]           # frame-dependent decision
    hard_t ~ Bernoulli(p_t) if training else p_t>=0.5 # Pick/Drop
    hard_0 = 1                                        # first frame forced PICK
    S = {t : hard_t = 1}
"""
import math

import torch
from torch import nn

from xmodaler.config import configurable

from . import SELECTOR_REGISTRY
from .base_selector import BaseSelector

__all__ = ["PickNetStyleSelector"]


@SELECTOR_REGISTRY.register()
class PickNetStyleSelector(BaseSelector):
    @configurable
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        init_pick_prob: float,
        force_first_pick: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.force_first_pick = force_first_pick

        self.gru = nn.GRUCell(input_dim, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        # initialize the PICK logit bias so the initial policy starts near
        # init_pick_prob (avoids degenerate all-drop early training)
        if init_pick_prob > 0.0 and init_pick_prob < 1.0:
            with torch.no_grad():
                self.mlp[-1].bias[1] = math.log(init_pick_prob / (1.0 - init_pick_prob))

        self.last_stats = None  # detached per-step stats for logging hooks

    @classmethod
    def from_config(cls, cfg):
        return {
            "input_dim": cfg.MODEL.VISUAL_EMBED.IN_DIM,
            "hidden_size": cfg.SELECTOR.HIDDEN_SIZE,
            "init_pick_prob": cfg.SELECTOR.INIT_PICK_PROB,
        }

    @classmethod
    def add_config(cls, cfg, tmp_cfg):
        pass

    def forward(self, feats, masks):
        B, T, D = feats.shape
        device = feats.device

        h = torch.zeros(B, self.hidden_size, device=device)
        probs = []
        for t in range(T):
            xt = feats[:, t]                       # (B, D)
            mt = masks[:, t]                       # (B,)
            h_new = self.gru(xt, h)                # sequential state
            logits = self.mlp(torch.cat([h_new, xt], dim=-1))
            p = logits.softmax(dim=-1)[:, 1]       # P(PICK)
            p = p * mt                             # invalid candidates: p = 0
            if self.force_first_pick and t == 0:
                p = torch.ones_like(p)
            # carry state only through valid candidates
            h = h_new * mt.unsqueeze(-1) + h * (1.0 - mt).unsqueeze(-1)
            probs.append(p)
        probs = torch.stack(probs, dim=1)          # (B, T)

        if self.training:
            # stochastic policy + straight-through estimator: forward uses the
            # hard 0/1 decision, gradients flow through probs
            hard = (torch.rand_like(probs) < probs).float()
            if self.force_first_pick:
                hard[:, 0] = 1.0
            hard_ste = probs + (hard - probs).detach()
        else:
            hard = (probs >= 0.5).float()
            if self.force_first_pick:
                hard[:, 0] = 1.0
            hard_ste = hard

        selection_mask = hard * masks                  # selected AND valid
        selected_features = feats * hard_ste.unsqueeze(-1)

        indices = []
        for b in range(B):
            idx = (selection_mask[b] > 0.5).nonzero(as_tuple=False)
            indices.append(idx.squeeze(-1).tolist())

        num_selected = selection_mask.sum(dim=1)       # (B,)
        num_candidates = masks.sum(dim=1)              # (B,)

        with torch.no_grad():
            ns = num_selected.float()
            nc = num_candidates.float().clamp(min=1)
            self.last_stats = {
                "avg_frames": ns.mean().item(),
                "min_frames": ns.min().item(),
                "max_frames": ns.max().item(),
                "median_frames": ns.median().item(),
                "selection_ratio": (ns / nc).mean().item(),
            }

        return {
            "selected_features": selected_features,
            "selection_mask": selection_mask,
            "selection_probs": probs,
            "selection_indices": indices,
            "num_selected": num_selected,
            "num_candidates": num_candidates,
            "stats": self.last_stats,
        }
