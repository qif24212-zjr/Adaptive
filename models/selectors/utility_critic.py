# -*- coding: utf-8 -*-
"""
Utility Critic for the Adaptive Selector.

    V_hat_t = critic(h_t)

estimates the future return G_t = sum_{k=t+1}^{N} r_k (gamma = 1.0) of the
state h_t BEFORE the action at step t — i.e. "how much reward is left if we
keep selecting". It is NOT a frame-scoring network.

    L_util = 1/2 * mean_t (V_hat_t - G_t)^2   over selection steps only
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["UtilityCritic"]


class UtilityCritic(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """hidden_states: (B, S, H) pre-update states -> (B, S, 1) values."""
        return self.mlp(hidden_states)

    def utility_loss(self, values: Tensor, returns: Tensor, valid_mask: Tensor) -> Tensor:
        """1/2 * mean over selection steps of (V_hat_t - G_t)^2."""
        diff = (values.squeeze(-1) - returns) * valid_mask
        n = valid_mask.sum().clamp(min=1)
        return 0.5 * (diff ** 2).sum() / n
