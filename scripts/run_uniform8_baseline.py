# -*- coding: utf-8 -*-
"""
Uniform-8 frame baseline on the CoCap frame-level path (CapERA test split).

    Video (real ERA mp4)
      -> uniform sample 8 frames (cv2, center-crop 224, ImageNet norm)
      -> CLIP ViT-B/16 (CoCap IFrameEncoder)   [8, 512]
      -> FrameCaptionHead (greedy decode)      -> caption

This verifies the NEW frame-level path end-to-end on real data. NOTE: the
caption head is pretrained-INITIALIZED only (no captioning training has run
yet), so metric scores are path-sanity numbers, not model quality.

Run:  python scripts/run_uniform8_baseline.py [--limit N] [--out experiments/uniform8_framecocap/pred_test.json]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from models.frame_cocap import build_frame_captioner, generate_caption, read_video_frames_cv2

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_video_index(video_root: Path) -> dict:
    """Map normalized basename ('Class_001.mp4', spaces stripped) -> path.
    Videos live in per-class subdirectories; the ERA zip also ships each
    video under two names ('X .mp4' and 'X.mp4')."""
    index = {}
    for p in video_root.rglob("*.mp4"):
        key = p.name.replace(" ", "")
        index[key] = p
    return index


def resolve_video_path(index: dict, file_name: str) -> Path:
    key = file_name.replace(" ", "")
    if key in index:
        return index[key]
    raise FileNotFoundError(file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="cap number of videos")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(f"[1] building frame captioner on {DEVICE} ...")
    model = build_frame_captioner(
        frame_encoder="clip_vitb16",
        clip_path="checkpoints/clip/ViT-B-16.pt",
    ).to(DEVICE).eval()

    print("[2] loading CapERA test split ...")
    ann_json = json.load(open("data/CapERA/captions_test.json"))
    video_root = Path("datasets/CapERA/videos/Videos/Test")
    refs = {}
    for a in ann_json["annotations"]:
        refs.setdefault(str(a["image_id"]), []).append(a["caption"])
    images = ann_json["images"][: args.limit] if args.limit else ann_json["images"]
    print(f"    {len(images)} videos, {len(refs)} with refs")

    print(f"[3] uniform-{args.num_frames} inference ...")
    video_index = build_video_index(video_root)
    print(f"    indexed {len(video_index)} video files")
    results, n_skipped = {}, 0
    t0 = time.time()
    for img in tqdm(images, desc="inference"):
        try:
            video_path = resolve_video_path(video_index, img["file_name"])
        except FileNotFoundError:
            n_skipped += 1
            continue
        frames = read_video_frames_cv2(str(video_path), n_frames=args.num_frames, sample="uniform")
        if frames is None:
            n_skipped += 1
            continue
        caption = generate_caption(model, frames=frames.unsqueeze(0).to(DEVICE))[0]
        results[str(img["id"])] = [{"sentence": caption}]
    print(f"    {len(results)}/{len(images)} done, {n_skipped} skipped, "
          f"{time.time() - t0:.1f}s")

    # show a few examples
    for i, img in enumerate(images[:3]):
        vid = str(img["id"])
        if vid in results:
            print(f"    [{img['file_name']}] pred: '{results[vid][0]['sentence']}'")
            print(f"                        gt:   '{refs[vid][0]}'")

    # save predictions
    out_file = args.out or f"experiments/uniform8_framecocap/pred_test_n{args.num_frames}.json"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({"version": "VERSION 1.0", "results": results}, f, indent=2)
    print(f"[4] predictions saved to {out_file}")

    print("[5] evaluation (pycocoevalcap vs 5 refs) ...")
    sys.path.insert(0, str(Path("third_party/CoCap").resolve()))
    from cocap.modeling.eval_captioning import evaluate
    refs_sub = {k: v for k, v in refs.items() if k in results}  # pycocoevalcap needs equal key sets
    metrics = evaluate(submission={"results": results}, reference=refs_sub)
    for k, v in metrics.items():
        print(f"    {k:8s} {v * 100:.2f}" if isinstance(v, float) else f"    {k:8s} {v}")
    print("NOTE: caption head is UNTRAINED (path sanity only).")


if __name__ == "__main__":
    main()
