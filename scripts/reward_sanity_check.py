# -*- coding: utf-8 -*-
"""
Adaptive Reward Sanity Check.

Does the frozen, trained Frame-CoCap captioner produce CIDEr rewards that
DISCRIMINATE between different frame subsets of the same video? If not, an
RL selector has no reward signal and training must not start.

For 20 CapERA val videos:
    candidate pool = uniform 32 frames (project protocol, unchanged)
    subsets (all from the SAME pool, features encoded once):
        All-32, Uniform-16/8/4/2, Random-8 x3 seeds, Random-4 x3 seeds
    -> frozen CLIP + frozen captioner -> greedy caption -> CIDEr (5 refs)

Incremental reward (10 videos): S1=[5,20] -> S2=[5,20,27] -> S3=[5,20,27,31]
    Q_t, dQ_t = Q_t - Q_{t-1} (Q_0 = 0), r_t = dQ_t - lambda_cost

Frame-location sensitivity (10 videos x 12 random 8-frame subsets):
    per-video CIDEr std across locations — is location actually informative?

Outputs: experiments/adaptive_reward_sanity/{results,summary,examples}.json

This script only MEASURES. It never trains anything and never modifies
captioner / selector / sampling / CoCap.
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "xmodaler"))

import torch
import yaml

from models.frame_cocap import build_frame_captioner, generate_caption, read_video_frames_cv2
from models.frame_cocap.frame_sampling import uniform_sample_indices
from models.selectors.caption_reward import CaptionReward

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_CHUNK = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adaptive_frame_pruning.yaml")
    parser.add_argument("--num-videos", type=int, default=20)
    parser.add_argument("--num-incremental", type=int, default=10)
    parser.add_argument("--num-location", type=int, default=10)
    parser.add_argument("--location-subsets", type=int, default=12)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    lambda_cost = cfg["reward"]["lambda_cost"]
    ckpt = cfg["captioner"]["checkpoint"]
    M = cfg["candidate_frames"]
    out_dir = PROJECT_ROOT / "experiments/adaptive_reward_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"lambda_cost = {lambda_cost} (from {args.config})")
    print(f"captioner checkpoint = {ckpt}")

    # ---- frozen captioner ----
    captioner = build_frame_captioner(frame_encoder=cfg["frame_encoder"],
                                      clip_path=cfg["clip_path"]).to(DEVICE).eval()
    state = torch.load(ckpt, map_location=DEVICE)
    captioner.load_state_dict(state["captioner"], strict=True)
    for p in captioner.parameters():
        p.requires_grad_(False)
    # corpus-level DF for meaningful per-sample CIDEr (SCST-style)
    corpus_refs = {}
    for split in ("train", "val", "test"):
        a = json.load(open(f"data/CapERA/captions_{split}.json"))
        for x in a["annotations"]:
            corpus_refs.setdefault(f"{split}_{x['image_id']}", []).append(x["caption"])
    reward = CaptionReward(mode="terminal", metric="cider",
                           lambda_cost=lambda_cost, max_selected_frames=M,
                           corpus_refs=corpus_refs)
    print(f"loaded captioner (epoch {state.get('epoch')}) | "
          f"CIDEr corpus DF over {len(corpus_refs)} reference sets")

    # ---- data: random CapERA val videos ----
    ann = json.load(open("data/CapERA/captions_val.json"))
    refs = {}
    for x in ann["annotations"]:
        refs.setdefault(x["image_id"], []).append(x["caption"])
    images = ann["images"]
    rng = random.Random(0)
    rng.shuffle(images)
    images = images[:args.num_videos]
    video_index = {}
    for p in Path("datasets/CapERA/videos/Videos/Tra").rglob("*.mp4"):
        video_index[p.name.replace(" ", "")] = p
    print(f"{len(images)} videos")

    def caption_for(feats_subset, indices):
        """Frozen features-only caption + CIDEr for one subset."""
        with torch.no_grad():
            mask = torch.ones(1, len(indices), dtype=torch.long, device=DEVICE)
            cap = generate_caption(captioner, visual_features=feats_subset, frame_mask=mask)[0]
        return cap

    results, examples = [], []
    incremental_records = []
    location_stats = []

    t0 = time.time()
    for v_idx, img in enumerate(images):
        vid = img["id"]
        refs_v = refs[vid]
        path = video_index[img["file_name"].replace(" ", "")]
        frames = read_video_frames_cv2(str(path), n_frames=M, sample="uniform")
        assert frames is not None and frames.shape[0] == M
        frames = frames.to(DEVICE)
        # encode the 32-pool ONCE (frozen CLIP); all subsets gather from it
        with torch.no_grad():
            pool = torch.cat([captioner.frame_encoder(frames[i:i + CLIP_CHUNK])
                              for i in range(0, M, CLIP_CHUNK)], dim=0)  # (32, 512)
        # cache captions for reused subsets (avoid duplicate decoding)
        cap_cache = {}

        def run_subset(name, idxs):
            idxs = sorted(idxs)
            key = (name, tuple(idxs))
            if key in cap_cache:
                return cap_cache[key]
            cap = caption_for(pool[idxs].unsqueeze(0), idxs)
            score = reward.cider_scores([cap], [refs_v])[0]
            cap_cache[key] = (cap, score)
            return cap, score

        # fixed subsets (record all32 too)
        cap_all, q_all = run_subset("all32", list(range(M)))
        results.append({
            "video_id": img["file_name"], "video_int_id": vid,
            "subset": "all32", "selected_indices": list(range(M)),
            "caption": cap_all, "cider": round(q_all, 3),
            "delta_cider_vs_all32": 0.0,
        })
        subset_defs = []
        for n in (16, 8, 4, 2):
            subset_defs.append((f"uniform{n}", uniform_sample_indices(M, n)))
        for n, seeds in ((8, (0, 1, 2)), (4, (0, 1, 2))):
            for s in seeds:
                rr = random.Random(1000 * v_idx + 37 * n + s)
                subset_defs.append((f"random{n}_seed{s}", rr.sample(range(M), n)))

        for name, idxs in subset_defs:
            cap, q = run_subset(name, idxs)
            results.append({
                "video_id": img["file_name"], "video_int_id": vid,
                "subset": name, "selected_indices": idxs,
                "caption": cap, "cider": round(q, 3),
                "delta_cider_vs_all32": round(q - q_all, 3),
            })

        # incremental rewards (paper convention: Q_0 = 0)
        if v_idx < args.num_incremental:
            q_prev = 0.0
            cur = []
            for step, add_idx in enumerate((5, 20, 27, 31), start=1):
                cur = sorted(cur + [add_idx])
                cap, q_t = run_subset(f"incremental{step}", cur)
                dq = q_t - q_prev
                r_t = dq - lambda_cost
                incremental_records.append({"delta_cider": dq, "reward": r_t,
                                            "cider": q_t})
                examples.append({
                    "video_id": img["file_name"], "step": step,
                    "selected_indices": cur, "caption": cap,
                    "ground_truth": refs_v[0], "CIDEr": round(q_t, 3),
                    "delta_CIDEr": round(dq, 3), "reward": round(r_t, 4),
                })
                q_prev = q_t

        # frame-LOCATION sensitivity: many random 8-frame subsets, same video
        if v_idx < args.num_location:
            loc_scores = []
            for s in range(args.location_subsets):
                rr = random.Random(7777 * v_idx + s)
                loc_scores.append(run_subset(f"loc{s}", rr.sample(range(M), 8))[1])
            location_stats.append({"video_id": img["file_name"], "scores": loc_scores,
                                   "std": statistics.pstdev(loc_scores),
                                   "min": min(loc_scores), "max": max(loc_scores)})

        print(f"[{v_idx + 1}/{len(images)}] {img['file_name']} q32={q_all:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- summary ----
    def group(name):
        vals = [r["cider"] for r in results if r["subset"] == name]
        return vals

    def mean_std(vals):
        return (round(statistics.mean(vals), 3), round(statistics.pstdev(vals), 3))

    summary = {"num_videos": len(images), "candidate_pool": M,
               "lambda_cost": lambda_cost}
    for name in ("all32", "uniform16", "uniform8", "uniform4", "uniform2"):
        m, s = mean_std(group(name))
        summary[f"mean_cider.{name}"] = m
        summary[f"std_cider.{name}"] = s
    for n in (8, 4):
        vals = [r["cider"] for r in results if r["subset"].startswith(f"random{n}")]
        m, s = mean_std(vals)
        summary[f"random{n}_mean"] = m
        summary[f"random{n}_std"] = s
    dq_vals = [r["delta_cider"] for r in incremental_records]
    r_vals = [r["reward"] for r in incremental_records]
    summary["mean_delta_cider"] = round(statistics.mean(dq_vals), 3)
    summary["median_delta_cider"] = round(statistics.median(dq_vals), 3)
    summary["std_delta_cider"] = round(statistics.pstdev(dq_vals), 3)
    summary["mean_reward"] = round(statistics.mean(r_vals), 4)
    summary["median_reward"] = round(statistics.median(r_vals), 4)
    summary["positive_reward_ratio"] = round(sum(1 for r in r_vals if r > 0) / len(r_vals), 3)
    # location sensitivity
    per_video_std = [s["std"] for s in location_stats]
    summary["location_mean_per_video_std"] = round(statistics.mean(per_video_std), 3)
    summary["location_std_of_std"] = round(statistics.pstdev(per_video_std), 3)
    summary["location_min_max_spread_mean"] = round(
        statistics.mean(s["max"] - s["min"] for s in location_stats), 3)
    # subset-size trend (is more frames monotonically better on average?)
    summary["trend_q32_q16_q8_q4_q2"] = [summary["mean_cider.all32"]] + \
        [summary[f"mean_cider.uniform{n}"] for n in (16, 8, 4, 2)]

    for path, obj in (("results.json", results), ("summary.json", summary),
                      ("examples.json", examples)):
        with open(out_dir / path, "w") as f:
            json.dump(obj, f, indent=2)

    print("\n===== SUMMARY =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\noutputs -> {out_dir}")


if __name__ == "__main__":
    main()
