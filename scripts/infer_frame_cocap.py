# -*- coding: utf-8 -*-
"""
Inference for the Frame-CoCap captioner checkpoint.

    checkpoint -> CapERA test videos -> 32 uniform frames -> CLIP (frozen)
    -> FrameCaptionHead (greedy) -> captions -> pycocoevalcap metrics

The checkpoint format is shared with the Adaptive pipeline
(policy_trainer.load_captioner_checkpoint), so the same file drops straight
into selector training via captioner.checkpoint.

Usage:
    python scripts/infer_frame_cocap.py \
        --config configs/frame_captioning_training.yaml \
        --checkpoint experiments/frame_cocap/checkpoints/best.pt \
        [--limit 100]
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "xmodaler"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "CoCap"))

import torch
import yaml

from models.frame_cocap import build_frame_captioner, generate_caption, read_video_frames_cv2
from cocap.modeling.eval_captioning import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/frame_captioning_training.yaml")
    parser.add_argument("--checkpoint", default="experiments/frame_cocap/checkpoints/best.pt")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg["output_dir"])
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    captioner = build_frame_captioner(frame_encoder=cfg["frame_encoder"],
                                      clip_path=cfg["clip_path"]).to(DEVICE).eval()
    state = torch.load(args.checkpoint, map_location=DEVICE)
    captioner.load_state_dict(state["captioner"], strict=True)
    print(f"[checkpoint] {args.checkpoint} (epoch {state.get('epoch')}, "
          f"stats={state.get('stats')})")

    ann = json.load(open("data/CapERA/captions_test.json"))
    refs = {}
    for x in ann["annotations"]:
        refs.setdefault(x["image_id"], []).append(x["caption"])
    images = ann["images"][:args.limit]
    video_root = Path(cfg["data"]["test_video_root"])
    video_index = {p.name.replace(" ", ""): p for p in video_root.rglob("*.mp4")}

    results, n_skip = {}, 0
    t0 = time.time()
    for img in images:
        path = video_index[img["file_name"].replace(" ", "")]
        frames = read_video_frames_cv2(str(path), n_frames=cfg["num_frames"], sample="uniform")
        if frames is None or frames.shape[0] != cfg["num_frames"]:
            n_skip += 1
            continue
        frames = frames.to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            feats = torch.cat([captioner.frame_encoder(frames[:, i:i + CHUNK].squeeze(0))
                               for i in range(0, cfg["num_frames"], CHUNK)], dim=0).unsqueeze(0)
            mask = torch.ones(1, cfg["num_frames"], dtype=torch.long, device=DEVICE)
            cap = generate_caption(captioner, visual_features=feats, frame_mask=mask)[0]
        results[str(img["id"])] = [{"sentence": cap}]

    out_file = out_dir / "predictions" / f"predictions_ckpt_epoch{state.get('epoch')}.json"
    with open(out_file, "w") as f:
        json.dump({"version": "VERSION 1.0", "results": results}, f, indent=2)
    print(f"[infer] {len(results)}/{len(images)} videos ({time.time() - t0:.0f}s), "
          f"{n_skip} skipped -> {out_file}")

    refs_sub = {str(k): v for k, v in refs.items() if str(k) in results}
    metrics = {k: round(v * 100, 2) for k, v in
               evaluate(submission={"results": results}, reference=refs_sub).items()
               if isinstance(v, float)}
    print("[metrics]")
    for k, v in metrics.items():
        print(f"    {k:8s} {v}")
    with open(out_dir / "predictions" / f"metrics_ckpt_epoch{state.get('epoch')}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[samples]")
    for img in images[:4]:
        print(f"    [{img['file_name']}] pred: '{results[str(img['id'])][0]['sentence']}'")
        print(f"                        gt:   '{refs[img['id']][0]}'")


if __name__ == "__main__":
    main()
