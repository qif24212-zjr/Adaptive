# -*- coding: utf-8 -*-
"""
Lightweight frame encoder for the Adaptive Selector front-end.

    RGB 224x224 -> small strided CNN -> global average pooling -> Linear -> 512

The whole point: the selector must see CHEAP per-frame features, so only the
few selected frames ever enter the heavy CLIP ViT-B/16. This net is ~57K
params / ~0.4 GFLOPs per frame vs CLIP ViT-B/16's ~86M params / ~17.6 GFLOPs.
No pretrained weights, no downloads: trained end-to-end by the selector's
policy-gradient signal (Phase 7+).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["LightweightFrameEncoder"]


class LightweightFrameEncoder(nn.Module):
    def __init__(self, feature_dim: int = 512, width: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        # 224 -> 112 -> 56 -> 28 -> 14, channels 3 -> 16 -> 32 -> 64 -> 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, width, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(width, feature_dim)

    def forward(self, frames: Tensor) -> Tensor:
        """frames: (B, 3, H, W) -> (B, feature_dim)."""
        x = self.pool(self.stem(frames)).flatten(1)
        return self.proj(x)
