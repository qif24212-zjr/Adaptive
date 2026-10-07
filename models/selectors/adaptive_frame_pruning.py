# -*- coding: utf-8 -*-
"""
Adaptive Frame Pruning — the real computational path of the paper.

    candidate RGB frames [B, M, 3, 224, 224]
        -> LightweightFrameEncoder                     [B, M, D]  (cheap)
        -> AdaptiveSelector (SELECT / STOP)            selected_indices, mask, counts
        -> gather RAW RGB frames                       [B, N_max, 3, 224, 224] + mask
        -> heavy CLIP ViT-B/16 encodes ONLY the valid frames
           (sum(counts) frames, never B*M)
        -> scatter into [B, N_max, D] padded features + mask
        -> existing Frame-CoCap CaptionHead            logits / captions

Gradient contract: the lightweight encoder + selector run WITH gradients
(trainable by the policy-gradient signal); the heavy CLIP encoder and the
caption head are FROZEN and run under torch.no_grad(). The captioner is
consumed in features-only mode, so its internal encoder never runs here.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.frame_cocap import generate_caption
from models.selectors.adaptive_selector import AdaptiveSelector
from models.selectors.lightweight_frame_encoder import LightweightFrameEncoder

__all__ = ["CountingEncoder", "gather_selected_frames", "AdaptiveFramePruning"]

CLIP_CHUNK = 32  # official CLIP ViT materializes attention-weight stacks;
# per-frame encoding is independent, so chunking is exact


class CountingEncoder(nn.Module):
    """Wraps the heavy frame encoder and counts frames it processes.

    Used to VERIFY efficiency: heavy CLIP frame count must equal the actual
    selected frame count, never B * M.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.frames_encoded = 0

    def reset(self) -> None:
        self.frames_encoded = 0

    def forward(self, frames: Tensor) -> Tensor:
        self.frames_encoded += int(frames.shape[0])
        return self.encoder(frames)


def gather_selected_frames(
        frames: Tensor,
        selected_indices: Tensor,
        selected_mask: Tensor,
):
    """Gather selected RAW frames per sample, pad to the batch max.

    Returns:
        selected_frames: (B, N_max, 3, H, W) zero-filled at padding slots
        mask:            (B, N_max) long (1 = valid; long to match the
                         FrameCaptionHead mask dtype contract)
    """
    B, M, C, H, W = frames.shape
    n_max = int(selected_mask.sum(dim=1).max())
    assert n_max >= 1
    idxs = selected_indices[:, :n_max].clamp(min=0)   # (B, N_max)
    mask = selected_mask[:, :n_max]                   # (B, N_max) bool
    gathered = frames[torch.arange(B, device=frames.device)[:, None], idxs]
    gathered = gathered * mask.view(B, n_max, 1, 1, 1)
    return gathered, mask.long()


class AdaptiveFramePruning(nn.Module):
    def __init__(
            self,
            lightweight_encoder: nn.Module,
            selector: AdaptiveSelector,
            heavy_encoder: nn.Module,   # heavy CLIP frame encoder (wrapped in CountingEncoder)
            captioner: nn.Module,       # FrameVideoCaptioner, features-only consumer
            encode_chunk: int = CLIP_CHUNK,
    ):
        super().__init__()
        self.lightweight_encoder = lightweight_encoder
        self.selector = selector
        self.heavy_encoder = heavy_encoder      # CountingEncoder instance
        self.captioner = captioner
        self.encode_chunk = encode_chunk

    def _encode_heavy(self, valid_frames: Tensor) -> Tensor:
        """Heavy CLIP over ONLY the given (valid) frames, chunked."""
        feats = torch.cat([
            self.heavy_encoder(valid_frames[i:i + self.encode_chunk])
            for i in range(0, valid_frames.shape[0], self.encode_chunk)
        ], dim=0)
        return feats

    def forward(self, candidate_frames: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            candidate_frames: (B, M, 3, H, W) normalized RGB candidate frames.

        Returns (in addition to the selector outputs):
            lightweight_features: (B, M, D)
            selected_frames:      (B, N_max, 3, H, W) gathered RAW frames
            selected_frame_mask:  (B, N_max) long
            heavy_features:       (B, N_max, D) CLIP features of selected frames,
                                  zero at padding slots
            heavy_frames_encoded: int = sum(selected_count), the ONLY frames
                                  the heavy CLIP ever saw
        """
        B, M, C, H, W = candidate_frames.shape

        # 1. lightweight features (trainable path)
        flat = candidate_frames.reshape(B * M, C, H, W)
        lw = self.lightweight_encoder(flat).reshape(B, M, -1)

        # 2. adaptive selection (trainable path)
        sel = self.selector(lw)

        # 3. gather RAW RGB frames (not features!)
        sel_frames, sel_frame_mask = gather_selected_frames(
            candidate_frames, sel["selected_indices"], sel["selected_mask"])

        # 4. heavy CLIP ONLY on valid frames (frozen, no grad)
        N_max = sel_frames.shape[1]
        flat_sel = sel_frames.reshape(B * N_max, C, H, W)
        valid_rows = sel_frame_mask.reshape(-1).bool()
        with torch.no_grad():
            heavy = self._encode_heavy(flat_sel[valid_rows])
        heavy_features = torch.zeros(B * N_max, heavy.shape[-1],
                                     dtype=heavy.dtype, device=heavy.device)
        heavy_features[valid_rows] = heavy
        heavy_features = (heavy_features.reshape(B, N_max, -1)
                          * sel_frame_mask.unsqueeze(-1))

        out = dict(sel)
        out.update({
            "lightweight_features": lw,
            "selected_frames": sel_frames,
            "selected_frame_mask": sel_frame_mask,
            "heavy_features": heavy_features,
            "heavy_frames_encoded": int(valid_rows.sum()),
        })
        return out

    @torch.no_grad()
    def generate(self, candidate_frames: Tensor) -> List[str]:
        """Select frames and greedily decode captions (frozen captioner)."""
        out = self.forward(candidate_frames)
        return generate_caption(
            self.captioner,
            visual_features=out["heavy_features"],
            frame_mask=out["selected_frame_mask"],
        )

    @torch.no_grad()
    def generate_from(self, out: Dict[str, Tensor]) -> List[str]:
        """Generate captions from a cached forward() output."""
        return generate_caption(
            self.captioner,
            visual_features=out["heavy_features"],
            frame_mask=out["selected_frame_mask"],
        )
