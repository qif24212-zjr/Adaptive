# -*- coding: utf-8 -*-
"""
Training utilities for the Frame-CoCap captioner (project-side; official
CoCap untouched).

    visual features [B, N, 512] + teacher-forced text tokens
        -> FrameCaptionHead -> prediction_scores [B, 77, 49408]
        -> cross-entropy (shifted labels, PAD ignored)

Checkpoint format {"captioner": state_dict, ...} is exactly what the
Adaptive pipeline's load_captioner_checkpoint hook (policy_trainer.py)
expects, so a trained captioner drops straight into selector training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.frame_cocap.frame_video_captioner import FrameVideoCaptioner, tokenize_captions

logger = logging.getLogger(__name__)

__all__ = [
    "captioning_loss",
    "CaptionerTrainer",
    "save_captioner_checkpoint",
    "load_captioner_checkpoint",
]


def captioning_loss(
        prediction_scores: Tensor,   # (B, 77, V)
        input_ids: Tensor,          # (B, 77) with BOS at 0
        label_smoothing: float = 0.1,
) -> Tensor:
    """Teacher-forced CE. Position i predicts token i+1 (official shift:
    labels = input_ids[1:] + PAD). PAD (0) targets are ignored."""
    labels = torch.cat([input_ids[:, 1:], torch.zeros(input_ids.shape[0], 1,
                                                      dtype=input_ids.dtype,
                                                      device=input_ids.device)], dim=1)
    return F.cross_entropy(
        prediction_scores.reshape(-1, prediction_scores.shape[-1]),
        labels.reshape(-1),
        ignore_index=0,  # PAD
        label_smoothing=label_smoothing,
    )


def build_optimizer(captioner: FrameVideoCaptioner, lr: float, lr_pretrained_embeddings: float,
                    freeze_frame_encoder: bool = True):
    """CLIP frame encoder frozen by default; the caption head trains with
    lr, the pretrained CLIP word embeddings + tied decoder with a smaller
    lr (mirrors the official CoCap clip_lr idea)."""
    for p in captioner.frame_encoder.parameters():
        p.requires_grad_(not freeze_frame_encoder)

    pretrained_keys = ("caption_head.cap_sa_decoder.word_embeddings",
                       "caption_head.prediction_head.decoder")
    pretrained, rest = [], []
    for name, p in captioner.caption_head.named_parameters():
        if name in pretrained_keys:
            pretrained.append(p)
        else:
            rest.append(p)
    return torch.optim.AdamW([
        {"params": rest, "lr": lr},
        {"params": pretrained, "lr": lr_pretrained_embeddings},
    ], weight_decay=0.01)


def save_captioner_checkpoint(path, captioner, optimizer, scheduler, epoch, config,
                              extra: Optional[Dict] = None):
    """Format compatible with policy_trainer.load_captioner_checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "captioner": captioner.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "config": config,
        "tokenizer": "clip_bpe_49408",  # vocab 49408, PAD 0, BOS 49406, EOS 49407
    }
    if extra:
        state["stats"] = extra
    torch.save(state, path)
    return path


def load_captioner_checkpoint(path, captioner, optimizer=None, scheduler=None, device=None):
    """Project-side loader (same file format as the Adaptive hook)."""
    state = torch.load(path, map_location=device or "cpu")
    captioner.load_state_dict(state["captioner"], strict=True)
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return state


class CaptionerTrainer:
    """Minimal supervised trainer for FrameVideoCaptioner."""

    def __init__(
            self,
            captioner: FrameVideoCaptioner,
            lr: float = 1e-4,
            lr_pretrained_embeddings: float = 1e-5,
            label_smoothing: float = 0.1,
            freeze_frame_encoder: bool = True,
    ):
        self.captioner = captioner
        self.label_smoothing = label_smoothing
        self.optimizer = build_optimizer(captioner, lr, lr_pretrained_embeddings,
                                         freeze_frame_encoder)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def train(self):
        self.captioner.train()

    def eval(self):
        self.captioner.eval()

    def train_step(self, visual_features: Tensor, frame_mask: Tensor,
                   captions: Sequence[str]) -> Tensor:
        """One supervised step: features + GT captions -> CE loss."""
        input_ids, input_mask = tokenize_captions(list(captions))
        input_ids = input_ids.to(visual_features.device)
        input_mask = input_mask.to(visual_features.device)
        prediction_scores = self.captioner(
            visual_features=visual_features, frame_mask=frame_mask,
            input_ids=input_ids, input_mask=input_mask)
        return captioning_loss(prediction_scores, input_ids,
                               label_smoothing=self.label_smoothing)

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def scheduler_step(self):
        self.scheduler.step()

    def save(self, path, epoch, config, extra=None):
        return save_captioner_checkpoint(path, self.captioner, self.optimizer,
                                         self.scheduler, epoch, config, extra)

    def load(self, path, device=None):
        return load_captioner_checkpoint(path, self.captioner, self.optimizer,
                                         self.scheduler, device)
