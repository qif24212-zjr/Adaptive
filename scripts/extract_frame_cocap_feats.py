# -*- coding: utf-8 -*-
"""
Pre-extract frozen CLIP ViT-B/16 features for Frame-CoCap captioner training.

Since the frame encoder is FROZEN during captioner training, features are
static — extract once to disk ((32, 512) fp32 per video) and train epochs
become cheap. Also used by the Adaptive pipeline's feature cache later.

Usage:
    python scripts/extract_frame_cocap_feats.py --split train [--split val] \
        [--limit N] [--features-dir features/frame_cocap/clip_vitb16]
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "xmodaler"))

import torch
from tqdm import tqdm

from models.frame_cocap import build_frame_captioner, read_video_frames_cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", default=["train"],
                        choices=["train", "val", "test"])
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--features-dir", default="features/frame_cocap/clip_vitb16")
    args = parser.parse_args()

    captioner = build_frame_captioner(frame_encoder="clip_vitb16",
                                      clip_path="checkpoints/clip/ViT-B-16.pt").to(DEVICE).eval()
    encoder = captioner.frame_encoder

    for split in args.split:
        ann = json.load(open(f"data/CapERA/captions_{split}.json"))
        video_root = Path("datasets/CapERA/videos/Videos") / ("Tra" if split != "test" else "Test")
        index = {}
        for p in video_root.rglob("*.mp4"):
            index[p.name.replace(" ", "")] = p
        out_dir = Path(args.features_dir) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        images = ann["images"][: args.limit] if args.limit else ann["images"]

        t0 = time.time()
        n_done = 0
        for img in tqdm(images, desc=f"extract {split}"):
            out_path = out_dir / f"{img['id']}.pt"
            if out_path.exists():
                continue
            path = index[img["file_name"].replace(" ", "")]
            frames = read_video_frames_cv2(str(path), n_frames=args.num_frames, sample="uniform")
            if frames is None or frames.shape[0] != args.num_frames:
                print(f"skip {img['file_name']}: {None if frames is None else frames.shape[0]} frames")
                continue
            frames = frames.to(DEVICE)
            feats = torch.cat([encoder(frames[i:i + CHUNK])
                               for i in range(0, frames.shape[0], CHUNK)], dim=0).cpu()
            torch.save(feats, out_path)
            n_done += 1
        print(f"[{split}] {n_done}/{len(images)} videos in {time.time() - t0:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
