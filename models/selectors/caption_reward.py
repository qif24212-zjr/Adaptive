# -*- coding: utf-8 -*-
"""
Caption-aware reward for the Adaptive Selector (paper definition).

Paper definition (caption_reward_mode = "full"):

    Q_t   = CIDEr(y_hat_t, y_gt)      # caption from the first t+1 selected frames
    dQ_t  = Q_t - Q_{t-1},  Q_0 = 0
    r_t   = dQ_t - lambda_cost        # per selection step
    r_STOP = 0
    R(tau) = sum_t r_t = Q_N - lambda_cost * N

Engineering acceleration (caption_reward_mode = "terminal"):

    exactly one captioner run on the FINAL selected subset, the entire
    R(tau) is placed on the last selection step:
    r_{N-1} = Q_N - lambda_cost * N,  r_t = 0 for t < N-1,  r_STOP = 0
    -> sum_t r_t is IDENTICAL to the paper formula, but no per-step
       caption decoding / CIDEr is needed.

The two modes are kept strictly separate in code and config; "terminal" is
the default for training efficiency, "full" exists to validate the formula.
CIDEr never backpropagates into the selector: captions are decoded under
torch.no_grad() by the frozen captioner.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.cider.cider_scorer import CiderScorer

logger = logging.getLogger(__name__)

__all__ = ["CaptionReward", "CorpusCider"]


class CorpusCider:
    """CIDEr-D with corpus-level statistics for per-sample scoring.

    pycocoevalcap's Cider computes document frequencies (DF) and the corpus
    size (ref_len) FROM THE INSTANCES PASSED IN ONE CALL — so scoring a
    single video (or a tiny batch) gives IDF=log(1)=0 and a score of
    exactly 0.0. For RL rewards this is broken: every step would see
    CIDEr=0 regardless of caption quality.

    Standard fix (SCST-style): precompute DF over the WHOLE reference
    corpus once, then score each sample with the corpus statistics.
    The CIDEr-D formula itself is unchanged.
    """

    def __init__(self, refs: Mapping[object, Sequence[str]]):
        self.corpus = CiderScorer()
        for rs in refs.values():
            self.corpus += (None, list(rs))   # cook refs only (no hyps)
        self.corpus.compute_doc_freq()
        self.df = self.corpus.document_frequency
        self.ref_len = float(np.log(len(self.corpus.crefs)))  # corpus size
        self.n = self.corpus.n
        self.sigma = self.corpus.sigma
        self.corpus_size = len(self.corpus.crefs)

    def score(self, hyp: str, refs: Sequence[str]) -> float:
        s = CiderScorer(test=hyp, refs=list(refs))
        s.document_frequency = self.df          # corpus-level DF
        s.ref_len = self.ref_len                # corpus-level size
        return float(s.compute_cider()[0])      # single-instance score


class CaptionReward:
    def __init__(
            self,
            mode: str = "terminal",   # "full" (paper per-step) | "terminal" (efficient)
            metric: str = "cider",
            lambda_cost: float = 0.01,
            gamma: float = 1.0,
            max_selected_frames: int = 16,
            corpus_refs: Optional[Mapping[object, Sequence[str]]] = None,
    ):
        assert mode in ("full", "terminal")
        assert metric == "cider", "only CIDEr implemented for now"
        self.mode = mode
        self.metric = metric
        self.lambda_cost = lambda_cost
        self.gamma = gamma
        self.max_selected_frames = max_selected_frames
        if corpus_refs is not None:
            # corpus-level DF for meaningful per-sample CIDEr (SCST-style).
            # Without it, tiny-batch CIDEr is structurally ~0 (idf=log(1)=0).
            self._corpus = CorpusCider(corpus_refs)
            self._cider = None
        else:
            self._corpus = None
            self._cider = Cider()  # legacy batch scorer (tiny corpora -> ~0)
            logger.warning("CaptionReward without corpus_refs: per-batch CIDEr "
                           "on tiny batches is structurally near zero; pass "
                           "corpus_refs (all reference captions) for a real "
                           "reward signal.")

    # ------------------------------------------------------------------ CIDEr
    def cider_scores(self, hyps: Sequence[str], refs: Sequence[Sequence[str]]) -> List[float]:
        """Per-sample CIDEr of `hyps` against reference caption sets `refs`."""
        if self._corpus is not None:
            return [self._corpus.score(h, list(r)) for h, r in zip(hyps, refs)]
        # legacy path (pycocoevalcap Cider expects plain caption strings)
        gts = {str(i): list(refs[i]) for i in range(len(hyps))}
        res = {str(i): [hyps[i]] for i in range(len(hyps))}
        _, scores = self._cider.compute_score(gts, res)
        return [float(s) for s in scores]

    # ------------------------------------------------------------- rewards
    def terminal_rewards(
            self,
            captions: Sequence[str],        # caption from the FINAL selection
            refs: Sequence[Sequence[str]],
            counts: Tensor,                # (B,) long
    ) -> Tensor:
        """r_{N-1} = CIDEr - lambda_cost * N, all other steps 0 (r_STOP = 0).
        sum_t r_t = Q_N - lambda_cost * N == the paper's R(tau)."""
        B = len(captions)
        scores = self.cider_scores(captions, refs)
        rewards = torch.zeros(B, self.max_selected_frames, device=counts.device)
        for b in range(B):
            n = int(counts[b].item())
            rewards[b, n - 1] = scores[b] - self.lambda_cost * n
        return rewards

    def full_step_rewards(
            self,
            step_captions: Sequence[Sequence[str]],   # step_captions[b][t]: caption of
            refs: Sequence[Sequence[str]],            # the first t+1 selected frames
            counts: Tensor,
    ) -> Tensor:
        """Paper formula: Q_t = CIDEr(y_hat_t), r_t = dQ_t - lambda_cost."""
        B = len(step_captions)
        rewards = torch.zeros(B, self.max_selected_frames, device=counts.device)
        for b in range(B):
            q_prev = 0.0
            for t in range(int(counts[b].item())):
                q_t = self.cider_scores([step_captions[b][t]], [refs[b]])[0]
                rewards[b, t] = (q_t - q_prev) - self.lambda_cost
                q_prev = q_t
        return rewards

    def compute_rewards(self, captions, refs, counts: Tensor) -> Tensor:
        if self.mode == "terminal":
            return self.terminal_rewards(captions, refs, counts)
        # "full": captions must be per-step captions (list of lists)
        return self.full_step_rewards(captions, refs, counts)


def discounted_returns(rewards: Tensor, gamma: float = 1.0) -> Tensor:
    """G_t = sum_{k=t+1}^{N-1} gamma^(k-t-1) * r_k over selection steps.

    Exact backward recurrence: G_{S-1} = 0, G_t = r_{t+1} + gamma * G_{t+1}.
    (The paper sets gamma = 1.0: all future rewards contribute directly.)"""
    S = rewards.shape[1]
    G = torch.zeros_like(rewards)
    for t in reversed(range(S - 1)):
        G[:, t] = rewards[:, t + 1] + gamma * G[:, t + 1]
    return G
