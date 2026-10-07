# -*- coding: utf-8 -*-
"""
Inference for the Adaptive Frame Pruning pipeline.

    python scripts/infer_adaptive_frame_pruning.py \
        --config configs/adaptive_frame_pruning.yaml \
        --checkpoint experiments/adaptive_frame_pruning/checkpoints/best.pt \
        [--limit 8]

Outputs under experiments/adaptive_frame_pruning/:
    predictions/predictions.json   generated captions
    selections/selection.json      per-video selected indices / counts / pruning ratios
    metrics.json                   stage timings + heavy-CLIP frame counts + scores

Stage timings separate the lightweight selector front-end from the heavy
CLIP encoder and the caption decoder. The heavy CLIP frame counter proves
the encoder only ever sees the selected frames.
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

from models.frame_cocap import build_frame_captioner, generate_caption, read_video_frames_cv2
from models.selectors.adaptive_frame_pruning import AdaptiveFramePruning, CountingEncoder, \
    gather_selected_frames
from models.selectors.adaptive_selector import AdaptiveSelector
from models.selectors.lightweight_frame_encoder import LightweightFrameEncoder
from models.selectors.policy_trainer import SelectorPolicyTrainer, load_captioner_checkpoint
from models.selectors.utility_critic import UtilityCritic

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adaptive_frame_pruning.yaml")
    parser.add_argument("--checkpoint", default="experiments/adaptive_frame_pruning/checkpoints/best.pt")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))

    out_dir = Path(cfg["output_dir"])
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "selections").mkdir(parents=True, exist_ok=True)

    # ---- build (inference: argmax policy) ----
    captioner = build_frame_captioner(frame_encoder=cfg["frame_encoder"],
                                      clip_path=cfg["clip_path"]).to(DEVICE).eval()
    for p in captioner.parameters():
        p.requires_grad_(False)
    if cfg["captioner"].get("checkpoint"):
        load_captioner_checkpoint(captioner, cfg["captioner"]["checkpoint"], DEVICE)

    selector = AdaptiveSelector(
        feature_dim=cfg["selector"]["feature_dim"], hidden_size=cfg["selector"]["hidden_size"],
        min_selected_frames=cfg["min_selected_frames"], max_selected_frames=cfg["max_selected_frames"],
        max_candidates=cfg["selector"]["max_candidates"], sample=False,  # argmax at inference
    ).to(DEVICE)
    lightweight = LightweightFrameEncoder(feature_dim=cfg["lightweight_encoder"]["feature_dim"],
                                          width=cfg["lightweight_encoder"]["width"]).to(DEVICE)
    pruning = AdaptiveFramePruning(
        lightweight_encoder=lightweight, selector=selector,
        heavy_encoder=CountingEncoder(captioner.frame_encoder), captioner=captioner)
    critic = UtilityCritic(input_dim=cfg["selector"]["hidden_size"],
                           hidden_dim=cfg["critic"]["hidden_dim"]).to(DEVICE)

    # load checkpoint (selector + lightweight + critic)
    state = torch.load(args.checkpoint, map_location=DEVICE)
    pruning.selector.load_state_dict(state["selector"])
    pruning.lightweight_encoder.load_state_dict(state["lightweight_encoder"])
    critic.load_state_dict(state["critic"])
    print(f"[checkpoint] loaded {args.checkpoint} (epoch {state.get('epoch')})")
    pruning.eval()

    # ---- data ----
    ann = json.load(open(cfg["data"]["annotation"]))
    refs = {}
    for x in ann["annotations"]:
        refs.setdefault(x["image_id"], []).append(x["caption"])
    images = ann["images"][:args.limit]
    video_index = {}
    for p in Path(cfg["data"]["video_root"]).rglob("*.mp4"):
        video_index[p.name.replace(" ", "")] = p

    M = cfg["candidate_frames"]
    predictions, selections = {}, []
    t_lw_total = t_clip_total = t_cap_total = 0.0
    heavy_frames_total = 0

    for img in images:
        vid = str(img["id"])
        frames = read_video_frames_cv2(str(video_index[img["file_name"].replace(" ", "")]),
                                       n_frames=M, sample="uniform").unsqueeze(0).to(DEVICE)

        # stage 1: lightweight features + selector
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            lw = pruning.lightweight_encoder(frames.reshape(-1, 3, 224, 224)).reshape(1, M, -1)
            sel = pruning.selector(lw)
        torch.cuda.synchronize(); t_lw = time.perf_counter() - t0

        # stage 2: gather + heavy CLIP (selected only)
        sel_frames, sel_mask = gather_selected_frames(
            frames, sel["selected_indices"], sel["selected_mask"])
        N_max = sel_frames.shape[1]
        flat = sel_frames.reshape(-1, 3, 224, 224)
        valid = sel_mask.reshape(-1).bool()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            heavy = torch.cat([pruning.heavy_encoder(flat[valid][i:i + 32])
                               for i in range(0, int(valid.sum()), 32)], dim=0)
            heavy_features = torch.zeros(flat.shape[0], heavy.shape[-1], device=DEVICE)
            heavy_features[valid] = heavy
            heavy_features = heavy_features.reshape(1, N_max, -1) * sel_mask.unsqueeze(-1)
        torch.cuda.synchronize(); t_clip = time.perf_counter() - t0

        # stage 3: caption generation
        torch.cuda.synchronize(); t0 = time.perf_counter()
        captions = generate_caption(captioner, visual_features=heavy_features, frame_mask=sel_mask)
        torch.cuda.synchronize(); t_cap = time.perf_counter() - t0

        t_lw_total += t_lw; t_clip_total += t_clip; t_cap_total += t_cap
        heavy_frames_total += int(valid.sum())

        count = int(sel["selected_count"].item())
        idxs = sel["selected_indices"][0, sel["selected_mask"][0]].tolist()
        predictions[vid] = [{"sentence": captions[0]}]
        selections.append({
            "video_id": img["file_name"],
            "selected_indices": idxs,
            "selected_count": count,
            "pruning_ratio": round(1 - count / M, 4),
        })

    n = len(images)
    # ---- outputs ----
    with open(out_dir / "predictions" / "predictions.json", "w") as f:
        json.dump({"version": "VERSION 1.0", "results": predictions}, f, indent=2)
    with open(out_dir / "selections" / "selection.json", "w") as f:
        json.dump(selections, f, indent=2)

    counts = [s["selected_count"] for s in selections]
    mean_sel = sum(counts) / max(len(counts), 1)
    # metrics (pycocoevalcap vs 5 refs)
    metrics = {"note": "captioner untrained -> scores are NOT meaningful (path sanity only)"}
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "CoCap"))
        from cocap.modeling.eval_captioning import evaluate
        refs_sub = {k: v for k, v in refs.items() if k in predictions}
        metrics.update({k: round(v * 100, 2) for k, v in evaluate(
            submission={"results": predictions}, reference=refs_sub).items() if isinstance(v, float)})
    except Exception as e:
        metrics["eval_error"] = str(e)

    metrics.update({
        "n_videos": n,
        "mean_selected_frames": round(mean_sel, 3),
        "min_selected_frames": min(counts) if counts else None,
        "max_selected_frames": max(counts) if counts else None,
        "retention_ratio": round(mean_sel / M, 4),
        "pruning_ratio": round(1 - mean_sel / M, 4),
        "heavy_clip_frames_encoded": heavy_frames_total,
        "candidate_frames_total": n * M,
        "time_lightweight_selector_s": round(t_lw_total, 3),
        "time_heavy_clip_s": round(t_clip_total, 3),
        "time_captioning_s": round(t_cap_total, 3),
        "time_total_s": round(t_lw_total + t_clip_total + t_cap_total, 3),
    })
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[selections]")
    for s in selections:
        print(f"  {s['video_id']}: count={s['selected_count']} "
              f"pruning_ratio={s['pruning_ratio']} indices={s['selected_indices']}")
    print(f"[efficiency] heavy CLIP encoded {heavy_frames_total} frames "
          f"of {n * M} candidates (would be {n * M} without pruning)")
    print(f"[timing] lightweight+selector {metrics['time_lightweight_selector_s']}s | "
          f"heavy CLIP {metrics['time_heavy_clip_s']}s | captioning {metrics['time_captioning_s']}s")
    print(f"[outputs] {out_dir / 'predictions/predictions.json'}, "
          f"{out_dir / 'selections/selection.json'}, {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
