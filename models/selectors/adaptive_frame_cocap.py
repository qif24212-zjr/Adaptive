# -*- coding: utf-8 -*-
"""
Integration wrapper: Adaptive Selector -> Frame-CoCap.

    candidate frames [B, M, 3, 224, 224]
        -> frame encoder (shared CLIP ViT-B/16)      [B, M, 512]
        -> AdaptiveSelector                          selected_indices/mask/count
        -> gather selected features                  [B, N_max, 512] + mask
        -> existing Frame-CoCap (FrameVideoCaptioner, features-only path)
        -> greedy caption generation

This module is a THIN glue layer only: it reuses
  - models/selectors/adaptive_selector.py   (AdaptiveSelector, unchanged)
  - models/frame_cocap/                     (encoder + caption head + generation,
                                             unchanged)

NOTE on compute order: for this integration smoke test the pipeline encodes
ALL M candidate frames with the heavy CLIP encoder first and only then runs
the selector. This proves the software pipeline, but does NOT yet provide
computational savings. The final paper version must move the selector onto
a lightweight feature so that only the selected frames go through the heavy
encoder ("lightweight selector features -> select -> heavy CLIP"). That
optimization is explicitly deferred to a later phase.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from models.frame_cocap import generate_caption
from models.selectors.adaptive_selector import AdaptiveSelector

__all__ = ["gather_selected", "AdaptiveFrameCoCap"]


def gather_selected(
        features: Tensor,
        selected_indices: Tensor,
        selected_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Gather selected frame features per sample, pad to the batch max.

    Args:
        features:         (B, M, D) candidate features
        selected_indices: (B, S) long, -1 padding (S = max_selected_frames)
        selected_mask:    (B, S) bool, prefix-true per row

    Returns:
        selected_features: (B, N_max, D) zero-filled where padded
        selected_mask:     (B, N_max) long (1 = valid) — long so it can be
                           concatenated with the text mask in FrameCaptionHead
    """
    B, M, D = features.shape
    n_max = int(selected_mask.sum(dim=1).max())
    assert n_max >= 1

    idxs = selected_indices[:, :n_max].clamp(min=0)   # (B, N_max); -1 -> 0 (masked below)
    mask = selected_mask[:, :n_max]                   # (B, N_max) bool

    gathered = features[torch.arange(B, device=features.device)[:, None], idxs]
    gathered = gathered * mask.unsqueeze(-1)          # zero the padding slots
    return gathered, mask.long()


class AdaptiveFrameCoCap(nn.Module):
    """Candidate frames -> select -> caption. Thin glue over existing modules."""

    def __init__(
            self,
            frame_encoder: nn.Module,
            selector: AdaptiveSelector,
            captioner: nn.Module,   # FrameVideoCaptioner (shares frame_encoder)
            encode_chunk: int = 32,  # per-frame encoding chunk (see forward)
    ):
        super().__init__()
        assert captioner.frame_encoder is frame_encoder, \
            "pass the SAME encoder object to captioner and this wrapper"
        self.frame_encoder = frame_encoder
        self.selector = selector
        self.captioner = captioner
        self.encode_chunk = encode_chunk

    def forward(self, candidate_frames: Tensor, candidate_mask: Optional[Tensor] = None) -> Dict:
        """
        Args:
            candidate_frames: (B, M, 3, H, W) RGB candidate frames
            candidate_mask:   (B, M) optional validity mask

        Returns:
            candidate_features:   (B, M, D)
            selected_indices:     (B, max_sel) long, -1 pad
            selected_mask:        (B, max_sel) bool      (selector step mask)
            selected_count:       (B,) long
            stop_step:            (B,) long
            selected_features:    (B, N_max, D) gathered, N_max = batch max count
            selected_feature_mask:(B, N_max) long         (caption-head mask)
        """
        B, M, C, H, W = candidate_frames.shape
        # Encode in chunks: the official CLIP ViT always materializes the
        # per-layer attention-weight stack (modified for visualization), so
        # very large per-frame batches OOM. Per-frame encoding is
        # independent, so chunking is exact.
        flat = candidate_frames.reshape(B * M, C, H, W)
        feats = torch.cat([
            self.frame_encoder(flat[i:i + self.encode_chunk])
            for i in range(0, B * M, self.encode_chunk)
        ], dim=0)
        candidate_features = feats.reshape(B, M, -1)

        sel = self.selector(candidate_features, frame_mask=candidate_mask)

        selected_features, selected_feature_mask = gather_selected(
            candidate_features, sel["selected_indices"], sel["selected_mask"])

        return {
            "candidate_features": candidate_features,
            "selected_indices": sel["selected_indices"],
            "selected_mask": sel["selected_mask"],
            "selected_count": sel["selected_count"],
            "stop_step": sel["stop_step"],
            "selected_features": selected_features,
            "selected_feature_mask": selected_feature_mask,
        }

    @torch.no_grad()
    def generate(self, candidate_frames: Tensor, candidate_mask: Optional[Tensor] = None) -> List[str]:
        """Select frames and greedily decode captions with the existing path."""
        out = self.forward(candidate_frames, candidate_mask=candidate_mask)
        return generate_caption(
            self.captioner,
            visual_features=out["selected_features"],
            frame_mask=out["selected_feature_mask"],
        )
