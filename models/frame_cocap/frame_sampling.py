Frame sampling and variable-length batching helpers for the frame path.

- uniform_sample_indices: same math as the official CoCap reader
  (sample_frames in video_readers.py, midpoint of equal intervals).
- collate_frames: pads variable-N frame tensors to N_max + builds the
  validity mask consumed by FrameCaptionHead.
- read_video_frames_cv2: decodes only the sampled frames of an mp4
  (mirrors official read_frames_cv2; cv2 only, no decord dependency).
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

__all__ = [
    "uniform_sample_indices",
    "collate_frames",
    "read_video_frames_cv2",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def uniform_sample_indices(total_frames: int, n: int) -> List[int]:
    """Uniformly sample n frame indices from [0, total_frames): midpoint of
    n equal intervals. Same math as official CoCap sample_frames(..., 'uniform')."""
    intervals = np.linspace(start=0, stop=total_frames, num=n + 1).astype(int)
    return [(int(intervals[i]) + int(intervals[i + 1])) // 2 for i in range(n)]


def collate_frames(frames_list: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """Pad variable-length frame tensors to the batch max.

    frames_list: list of (N_i, 3, H, W) float tensors, N_i variable
    returns: frames (B, N_max, 3, H, W) zero-padded, mask (B, N_max) long
    """
    B = len(frames_list)
    C, H, W = frames_list[0].shape[-3:]
    n_max = max(f.shape[0] for f in frames_list)
    frames = torch.zeros(B, n_max, C, H, W, dtype=frames_list[0].dtype,
                         device=frames_list[0].device)
    mask = torch.zeros(B, n_max, dtype=torch.long, device=frames_list[0].device)
    for i, f in enumerate(frames_list):
        frames[i, :f.shape[0]] = f
        mask[i, :f.shape[0]] = 1
    return frames, mask


def read_video_frames_cv2(
        video_path: str,
        n_frames: int,
        sample: str = "uniform",
        size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
) -> Optional[Tensor]:
    """Decode `n_frames` sampled frames of a video as (N, 3, 224, 224).

    Mirrors official read_frames_cv2 (video_readers.py) + the official
    CenterCrop/normalize pipeline, without the compressed-domain reader.
    Returns None if the video cannot be opened/decoded.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if vlen <= 0:
        cap.release()
        return None

    if sample == "uniform":
        idxs = uniform_sample_indices(vlen, n_frames)
    elif sample == "rand":
        idxs = sorted(random.sample(range(vlen), min(n_frames, vlen)))
    else:
        raise ValueError(f"unknown sample mode: {sample}")

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None

    # center crop to square, then resize (official pipeline: CenterCrop(224))
    out = []
    for f in frames:
        h, w = f.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        f = f[y0:y0 + s, x0:x0 + s]
        f = cv2.resize(f, size, interpolation=cv2.INTER_LINEAR)
        out.append(f)

    t = torch.from_numpy(np.stack(out)).float() / 255.0  # (N, H, W, 3)
    t = t.permute(0, 3, 1, 2)  # (N, 3, H, W)
    if normalize:
        mean = t.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = t.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        t = (t - mean) / std
    return t
