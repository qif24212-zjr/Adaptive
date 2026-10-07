# -*- coding: utf-8 -*-
"""
Evaluate ONE captioner checkpoint with the fixed progression protocol:
    val metrics (74 videos) + frame sensitivity (20 videos, fixed subsets)
and append the entry to experiments/frame_cocap/captioner_progression.json.

Usage:
    python scripts/eval_captioner_progression.py \
        --checkpoint experiments/frame_cocap/checkpoints/epoch_05.pt --epoch 5
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "xmodaler"))

import torch

from models.frame_cocap import build_frame_captioner
from models.frame_cocap.sensitivity_eval import (
    build_corpus_reward, build_video_index, eval_val_metrics,
    evaluate_frame_sensitivity, load_sensitivity_images,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--out", default="experiments/frame_cocap/captioner_progression.json")
    args = parser.parse_args()

    captioner = build_frame_captioner(frame_encoder="clip_vitb16",
                                      clip_path="checkpoints/clip/ViT-B-16.pt").to(DEVICE).eval()
    state = torch.load(args.checkpoint, map_location=DEVICE)
    captioner.load_state_dict(state["captioner"], strict=True)
    for p in captioner.parameters():
        p.requires_grad_(False)
    print(f"[checkpoint] {args.checkpoint} (saved epoch {state.get('epoch')})")

    # ---- val metrics (fixed 74-video val set, cached features) ----
    val_ann = json.load(open("data/CapERA/captions_val.json"))
    val_refs = {}
    for x in val_ann["annotations"]:
        val_refs.setdefault(x["image_id"], []).append(x["caption"])
    metrics = eval_val_metrics(captioner, val_ann["images"], val_refs,
                               "features/frame_cocap/clip_vitb16/val", DEVICE)
    print(f"[val] {metrics}")

    # ---- frame sensitivity (fixed 20 videos, fixed subsets) ----
    images, refs = load_sensitivity_images(20)
    reward = build_corpus_reward()
    video_index = build_video_index()
    stats = evaluate_frame_sensitivity(captioner, images, refs, reward, video_index,
                                       device=DEVICE, verbose=True)
    print(f"[sensitivity] {json.dumps(stats, indent=2)}")

    entry = {
        "epoch": args.epoch,
        "checkpoint": args.checkpoint,
        "val_cider": metrics.get("CIDEr"),
        "val_bleu4": metrics.get("Bleu_4"),
        "val_meteor": metrics.get("METEOR"),
        "val_rougeL": metrics.get("ROUGE_L"),
        **stats,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entries = json.load(open(out_path)) if out_path.exists() else []
    entries = [e for e in entries if e["epoch"] != args.epoch]
    entries.append(entry)
    entries.sort(key=lambda e: e["epoch"])
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
