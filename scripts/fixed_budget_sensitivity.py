# -*- coding: utf-8 -*-
"""
Fixed-budget captioning sensitivity diagnostic.

Compare three frozen captioners (epoch_10 / epoch_20 / best) on the SAME
32 CapERA train videos (seed 42), with fixed frame budgets N in
{2, 4, 8, 16, 32} (temporal-order-preserving uniform subsets of the
32-frame candidate pool) plus location subsets (12 each for N=4, N=8, same
locations across captioners).

Pure evaluation: no selector, no training, no optimizer. CIDEr uses the
SAME CorpusCider corpus statistics as the reward sanity check / RL runs.

Outputs -> experiments/fixed_budget_sensitivity/{results,summary,summary.csv,examples}.json

Run:
    python scripts/fixed_budget_sensitivity.py
"""

import csv
import json
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "xmodaler"))

import torch

from models.frame_cocap import build_frame_captioner, generate_caption
from models.frame_cocap.frame_sampling import uniform_sample_indices
from models.frame_cocap.sensitivity_eval import build_corpus_reward

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 32
BUDGETS = (2, 4, 8, 16, 32)
LOCATION_NS = (4, 8)
LOCATION_SUBSETS = 12
NUM_VIDEOS = 32
SEED = 42

CAPTIONERS = [
    ("epoch10", "experiments/frame_cocap/checkpoints/epoch_10.pt"),
    ("epoch20", "experiments/frame_cocap/checkpoints/epoch_20.pt"),
    ("best", "experiments/frame_cocap/checkpoints/best.pt"),
]
FRAMES_CACHE = Path("features/frame_cocap/frames/train")


def main():
    out_dir = PROJECT_ROOT / "experiments/fixed_budget_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"seed={SEED}, videos={NUM_VIDEOS}, budgets={BUDGETS}, "
          f"location N={LOCATION_NS} x {LOCATION_SUBSETS}")

    # ---- data: same 32 train videos for all captioners (seed 42) ----
    ann = json.load(open("data/CapERA/captions_train.json"))
    refs = {}
    for x in ann["annotations"]:
        refs.setdefault(x["image_id"], []).append(x["caption"])
    images = [img for img in ann["images"] if (FRAMES_CACHE / f"{img['id']}.pt").exists()]
    rng = random.Random(SEED)
    rng.shuffle(images)
    images = images[:NUM_VIDEOS]
    print(f"videos: {len(images)} (train split, all with 32 cached frames)")

    # ---- frozen captioners; assert identical frozen CLIP encoders ----
    captioners = {}
    for name, path in CAPTIONERS:
        c = build_frame_captioner(frame_encoder="clip_vitb16",
                                  clip_path="checkpoints/clip/ViT-B-16.pt").to(DEVICE).eval()
        state = torch.load(path, map_location=DEVICE)
        c.load_state_dict(state["captioner"], strict=True)
        for p in c.parameters():
            p.requires_grad_(False)
        captioners[name] = (c, state.get("epoch"))
        print(f"captioner {name}: {path} (saved epoch {state.get('epoch')})")
    enc0 = captioners["epoch10"][0].frame_encoder.state_dict()
    for name in ("epoch20", "best"):
        enc = captioners[name][0].frame_encoder.state_dict()
        assert all(torch.equal(enc0[k].cpu(), enc[k].cpu()) for k in enc0), \
            f"{name} frame encoder differs!"
    print("all three captioners share the identical frozen CLIP frame encoder "
          "-> encode each 32-pool ONCE")

    corpus_reward = build_corpus_reward()
    corpus_cider = corpus_reward._corpus

    # ---- per-video subset definitions (same across captioners) ----
    video_subsets = []   # (img, refs_v, pool_features, budget_subsets, loc_subsets)
    for v_idx, img in enumerate(images):
        frames = torch.load(FRAMES_CACHE / f"{img['id']}.pt", map_location=DEVICE)
        with torch.no_grad():
            pool = torch.cat([captioners["epoch10"][0].frame_encoder(frames[i:i + CHUNK])
                              for i in range(0, 32, CHUNK)], dim=0)  # (32, 512)
        budget_subsets = {n: uniform_sample_indices(32, n) for n in BUDGETS}
        vrng = random.Random(4242 + v_idx)
        loc_subsets = {}
        for n in LOCATION_NS:
            loc_subsets[n] = [sorted(vrng.sample(range(32), n))
                              for _ in range(LOCATION_SUBSETS)]
        video_subsets.append((img, refs[img["id"]], pool, budget_subsets, loc_subsets))
        if (v_idx + 1) % 8 == 0:
            print(f"  pools encoded: {v_idx + 1}/{len(images)}", flush=True)

    # ---- evaluate ----
    results = []
    per_cap = {name: {"budgets": {n: [] for n in BUDGETS},
                      "deltas": {}, "loc_n4": [], "loc_n8": [],
                      "unique": []}
               for name, _ in CAPTIONERS}

    def caption_cider(captioner, pool, idxs, refs_v):
        with torch.inference_mode():
            mask = torch.ones(1, len(idxs), dtype=torch.long, device=DEVICE)
            cap = generate_caption(captioner, visual_features=pool[idxs].unsqueeze(0),
                                   frame_mask=mask)[0]
        return cap, corpus_cider.score(cap, refs_v)

    for v_idx, (img, refs_v, pool, budget_subsets, loc_subsets) in enumerate(video_subsets):
        for name, _ in CAPTIONERS:
            c = captioners[name][0]
            caps_this = {}
            for n in BUDGETS:
                cap, score = caption_cider(c, pool, budget_subsets[n], refs_v)
                caps_this[f"b{n}"] = (cap, score)
                per_cap[name]["budgets"][n].append(score)
                results.append({"video_id": img["file_name"], "video_int_id": img["id"],
                                "captioner": name, "budget": n,
                                "selected_indices": budget_subsets[n],
                                "caption": cap, "cider": round(score, 3)})
            # per-video deltas (same video, same captioner)
            for a, b in zip(BUDGETS[:-1], BUDGETS[1:]):
                per_cap[name]["deltas"].setdefault(f"{a}->{b}", []).append(
                    caps_this[f"b{b}"][1] - caps_this[f"b{a}"][1])
            # location subsets (N=4, N=8)
            for n in LOCATION_NS:
                scores = [caption_cider(c, pool, idxs, refs_v)[1]
                          for idxs in loc_subsets[n]]
                per_cap[name][f"loc_n{n}"].append(statistics.pstdev(scores))
                for s, (idxs, sc) in enumerate(zip(loc_subsets[n], scores)):
                    results.append({"video_id": img["file_name"], "video_int_id": img["id"],
                                    "captioner": name, "budget": f"loc{n}_{s}",
                                    "selected_indices": idxs,
                                    "cider": round(sc, 3)})
            # unique captions over all 5 budgets + 24 locations
            uniq = len({v[0] for v in caps_this.values()} |
                       {caption_cider(c, pool, idxs, refs_v)[0]
                        for n in LOCATION_NS for idxs in loc_subsets[n]})
            per_cap[name]["unique"].append(uniq)
        print(f"  [{v_idx + 1}/{len(images)}] {img['file_name']} done", flush=True)

    # ---- summary ----
    def stats(vals):
        vals = [v for v in vals]
        n = max(len(vals), 1)
        return {
            "mean": round(statistics.mean(vals), 3),
            "median": round(statistics.median(vals), 3),
            "std": round(statistics.pstdev(vals), 3),
            "positive_ratio": round(sum(1 for v in vals if v > 1e-9) / n, 3),
            "zero_ratio": round(sum(1 for v in vals if abs(v) <= 1e-9) / n, 3),
            "negative_ratio": round(sum(1 for v in vals if v < -1e-9) / n, 3),
        }

    summary = {"protocol": {
        "seed": SEED, "num_videos": NUM_VIDEOS, "budgets": list(BUDGETS),
        "location_n": list(LOCATION_NS), "location_subsets": LOCATION_SUBSETS,
        "pool": "uniform 32 candidates, temporal-order-preserving uniform subsets",
        "cider": "CorpusCider (corpus DF over all CapERA refs)",
        "captions": {name: f"{path} (saved epoch {epoch})"
                     for (name, path), (_, epoch) in
                     zip(CAPTIONERS, [(name, captioners[name][1]) for name, _ in CAPTIONERS])},
    }}
    for name, _ in CAPTIONERS:
        p = per_cap[name]
        entry = {
            "cidr_at": {str(n): round(statistics.mean(p["budgets"][n]), 3) for n in BUDGETS},
            "delta": {k: stats(v) for k, v in p["deltas"].items()},
            "location": {
                "mean_std": round(statistics.mean(p["loc_n4"] + p["loc_n8"]), 4),
                "median_std": round(statistics.median(p["loc_n4"] + p["loc_n8"]), 4),
                "nonzero_fraction": round(
                    sum(1 for s in p["loc_n4"] + p["loc_n8"] if s > 1e-9)
                    / max(len(p["loc_n4"] + p["loc_n8"]), 1), 3),
                "mean_std_n4": round(statistics.mean(p["loc_n4"]), 4),
                "mean_std_n8": round(statistics.mean(p["loc_n8"]), 4),
            },
            "unique_captions_per_video": round(statistics.mean(p["unique"]), 2),
        }
        summary[name] = entry

    # ---- csv ----
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["captioner", "cidr2", "cidr4", "cidr8", "cidr16", "cidr32",
                    "d2_4", "d4_8", "d8_16", "d16_32",
                    "loc_std", "unique_captions", "pos_ratio_all", "zero_ratio_all"])
        for name, _ in CAPTIONERS:
            e = summary[name]
            pos = statistics.mean([e["delta"][k]["positive_ratio"] for k in e["delta"]])
            zero = statistics.mean([e["delta"][k]["zero_ratio"] for k in e["delta"]])
            w.writerow([name] + [e["cidr_at"][str(n)] for n in BUDGETS] +
                       [e["delta"][k]["mean"] for k in ("2->4", "4->8", "8->16", "16->32")] +
                       [e["location"]["mean_std"], e["unique_captions_per_video"],
                        round(pos, 3), round(zero, 3)])

    # ---- examples (10 representative videos) ----
    chosen = images[:10]
    examples = []
    for img in chosen:
        ex = {"video_id": img["file_name"], "ground_truth": refs[img["id"]]}
        for name, _ in CAPTIONERS:
            ex[name] = {}
            for r in results:
                if r["video_id"] == img["file_name"] and r["captioner"] == name \
                        and r["budget"] in (2, 8, 32):
                    ex[name][f"budget{r['budget']}"] = {
                        "caption": r["caption"], "cider": r["cider"]}
        examples.append(ex)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "examples.json", "w") as f:
        json.dump(examples, f, indent=2)
    print(f"[outputs] {out_dir}")

    # ---- print the tables ----
    print("\n| Captioner | CIDEr@2 | CIDEr@4 | CIDEr@8 | CIDEr@16 | CIDEr@32 |")
    print("| --------- | ------: | ------: | ------: | -------: | -------: |")
    for name, _ in CAPTIONERS:
        e = summary[name]
        print(f"| {name:9s} | " + " | ".join(
            f"{e['cidr_at'][str(n)]:>7.3f}" for n in BUDGETS) + " |")
    print("\n| Captioner | 2->4 | 4->8 | 8->16 | 16->32 | (mean per-video dCIDEr)")
    print("| --------- | --: | --: | ---: | ----: |")
    for name, _ in CAPTIONERS:
        e = summary[name]
        print(f"| {name:9s} | " + " | ".join(
            f"{e['delta'][k]['mean']:>4.3f}" for k in ("2->4", "4->8", "8->16", "16->32")) + " |")
    print("\n| Captioner | Location STD | Unique Captions | Pos d Ratio | Zero d Ratio |")
    print("| --------- | -----------: | --------------: | ----------: | -----------: |")
    for name, _ in CAPTIONERS:
        e = summary[name]
        pos = statistics.mean([e["delta"][k]["positive_ratio"] for k in e["delta"]])
        zero = statistics.mean([e["delta"][k]["zero_ratio"] for k in e["delta"]])
        print(f"| {name:9s} | {e['location']['mean_std']:>13.4f} | "
              f"{e['unique_captions_per_video']:>14.2f} | {pos:>10.3f} | {zero:>11.3f} |")


if __name__ == "__main__":
    main()
