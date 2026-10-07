# -*- coding: utf-8 -*-
"""
Predecode CapERA train videos into cached frame tensors (EXACT normalized
float32 form the model sees) so formal selector training does not spend
~45 min/epoch inside cv2. Pure engineering cache — the sampling protocol
(read_video_frames_cv2, uniform 32, center-crop 224, ImageNet norm) is
unchanged and the trainer verifies each cached tensor has 32 frames.

Usage:
    python scripts/predecode_capera_frames.py [--workers 8] [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "xmodaler"))

import joblib
import torch

from models.frame_cocap.frame_sampling import read_video_frames_cv2

OUT_DIR = Path("features/frame_cocap/frames/train")


def decode_one(args):
    img, video_index = args
    out_path = OUT_DIR / f"{img['id']}.pt"
    if out_path.exists():
        return None
    path = video_index[img["file_name"].replace(" ", "")]
    f = read_video_frames_cv2(str(path), n_frames=32, sample="uniform")
    if f is None or f.shape[0] != 32:
        return ("short", img["file_name"])
    torch.save(f, out_path)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ann = json.load(open("data/CapERA/captions_train.json"))
    images = ann["images"][: args.limit] if args.limit else ann["images"]
    video_index = {}
    for p in Path("datasets/CapERA/videos/Videos/Tra").rglob("*.mp4"):
        video_index[p.name.replace(" ", "")] = p
    print(f"[predecode] {len(images)} train videos, {args.workers} workers")

    results = joblib.Parallel(n_jobs=args.workers, backend="threading")(
        joblib.delayed(decode_one)((img, video_index)) for img in images)
    n_short = sum(1 for r in results if r is not None)
    n_done = len([p for p in OUT_DIR.glob("*.pt")])
    print(f"[predecode] done: {n_done} cached, {n_short} short videos skipped")


if __name__ == "__main__":
    main()
