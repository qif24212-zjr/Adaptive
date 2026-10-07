# -*- coding: utf-8 -*-
"""
Frame-sensitivity evaluation protocol (SHARED, fixed).

This is the exact protocol from scripts/reward_sanity_check.py (2026-09-30),
extracted verbatim so every captioner checkpoint is evaluated identically:

    - 20 CapERA val videos, selected by shuffling val images with
      random.Random(0) and taking the first 20 (SAME videos every run)
    - candidate pool = uniform 32 frames (project protocol, unchanged)
    - pool CLIP features encoded ONCE per video; every subset gathers from it
    - subsets: All-32, Uniform-16/8/4/2 (uniform_sample_indices math),
      Random-8 x3, Random-4 x3 (rng 1000*v_idx + 37*n + seed)
    - incremental: [5,20] -> [5,20,27] -> [5,20,27,31] (Q_0 = 0)
    - location: 10 videos x 12 random 8-frame subsets (rng 7777*v_idx + s)
    - per-sample CIDEr with corpus-level DF (CorpusCider)

Do NOT change this module without re-baselining: checkpoint comparisons
depend on the protocol staying identical.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from models.frame_cocap.frame_sampling import uniform_sample_indices
from models.frame_cocap.frame_video_captioner import generate_caption
from models.selectors.caption_reward import CaptionReward

CLIP_CHUNK = 32

# fixed subset protocol
UNIFORM_NS = (16, 8, 4, 2)
RANDOM_SPECS = ((8, (0, 1, 2)), (4, (0, 1, 2)))
INCREMENTAL_ADDS = (5, 20, 27, 31)
LOCATION_N = 8
LOCATION_SUBSETS = 12


def build_corpus_reward(lambda_cost: float = 0.01) -> CaptionReward:
    """CaptionReward with corpus-level DF over all CapERA refs."""
    corpus_refs = {}
    for split in ("train", "val", "test"):
        a = json.load(open(f"data/CapERA/captions_{split}.json"))
        for x in a["annotations"]:
            corpus_refs.setdefault(f"{split}_{x['image_id']}", []).append(x["caption"])
    return CaptionReward(mode="terminal", metric="cider", lambda_cost=lambda_cost,
                         max_selected_frames=32, corpus_refs=corpus_refs)


def load_sensitivity_images(num_videos: int = 20):
    """The SAME 20 val videos every run (random.Random(0) shuffle)."""
    ann = json.load(open("data/CapERA/captions_val.json"))
    refs = {}
    for x in ann["annotations"]:
        refs.setdefault(x["image_id"], []).append(x["caption"])
    images = ann["images"]
    rng = random.Random(0)
    rng.shuffle(images)
    return images[:num_videos], refs


def build_video_index(root: str = "datasets/CapERA/videos/Videos/Tra") -> Dict[str, Path]:
    index = {}
    for p in Path(root).rglob("*.mp4"):
        index[p.name.replace(" ", "")] = p
    return index


def subset_defs_for(v_idx: int, M: int = 32):
    """Identical to reward_sanity_check.py: uniform + seeded random subsets."""
    defs = []
    for n in UNIFORM_NS:
        defs.append((f"uniform{n}", uniform_sample_indices(M, n)))
    for n, seeds in RANDOM_SPECS:
        for s in seeds:
            rr = random.Random(1000 * v_idx + 37 * n + s)
            defs.append((f"random{n}_seed{s}", rr.sample(range(M), n)))
    return defs


def evaluate_frame_sensitivity(
        captioner,
        images: Sequence[Dict],
        refs: Dict,
        reward: CaptionReward,
        video_index: Dict[str, Path],
        num_incremental: int = 10,
        num_location: int = 10,
        device=None,
        verbose: bool = False,
) -> Dict:
    """Run the fixed subset protocol; returns the progression metrics."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = 32

    subset_scores = {f"uniform{n}": [] for n in UNIFORM_NS}
    subset_scores["all32"] = []
    random_scores = {8: [], 4: []}
    delta_qs: List[float] = []
    per_video_loc_std: List[float] = []
    unique_per_video: List[int] = []

    for v_idx, img in enumerate(images):
        vid = img["id"]
        refs_v = refs[vid]
        path = video_index[img["file_name"].replace(" ", "")]

        from models.frame_cocap.frame_sampling import read_video_frames_cv2
        frames = read_video_frames_cv2(str(path), n_frames=M, sample="uniform")
        assert frames is not None and frames.shape[0] == M, img["file_name"]
        frames = frames.to(device)
        with torch.no_grad():
            pool = torch.cat([captioner.frame_encoder(frames[i:i + CLIP_CHUNK])
                              for i in range(0, M, CLIP_CHUNK)], dim=0)  # (32, 512)

        cap_cache = {}

        def run_subset(idxs):
            idxs = sorted(idxs)
            key = tuple(idxs)
            if key in cap_cache:
                return cap_cache[key]
            with torch.no_grad():
                mask = torch.ones(1, len(idxs), dtype=torch.long, device=device)
                cap = generate_caption(captioner, visual_features=pool[idxs].unsqueeze(0),
                                       frame_mask=mask)[0]
            score = reward.cider_scores([cap], [refs_v])[0]
            cap_cache[key] = (cap, score)
            return cap, score

        # fixed subsets
        all_caps = []
        _, q_all = run_subset(list(range(M)))
        all_caps.append(run_subset(list(range(M)))[0])
        subset_scores["all32"].append(q_all)

        for name, idxs in subset_defs_for(v_idx, M):
            cap, q = run_subset(idxs)
            all_caps.append(cap)
            for n in UNIFORM_NS:
                if name == f"uniform{n}":
                    subset_scores[f"uniform{n}"].append(q)
            for n in (8, 4):
                if name.startswith(f"random{n}"):
                    random_scores[n].append(q)

        unique_per_video.append(len(set(all_caps)))

        # incremental rewards (Q_0 = 0)
        if v_idx < num_incremental:
            q_prev = 0.0
            cur = []
            for add_idx in INCREMENTAL_ADDS:
                cur = sorted(cur + [add_idx])
                _, q_t = run_subset(cur)
                delta_qs.append(q_t - q_prev)
                q_prev = q_t

        # frame-LOCATION sensitivity
        if v_idx < num_location:
            loc = []
            for s in range(LOCATION_SUBSETS):
                rr = random.Random(7777 * v_idx + s)
                loc.append(run_subset(rr.sample(range(M), LOCATION_N))[1])
            per_video_loc_std.append(statistics.pstdev(loc))

        if verbose:
            print(f"  sensitivity [{v_idx + 1}/{len(images)}] {img['file_name']} "
                  f"q32={q_all:.2f}", flush=True)

    stats = {
        "subset_cider": {
            "all32": round(statistics.mean(subset_scores["all32"]), 3),
            **{f"uniform{n}": round(statistics.mean(subset_scores[f"uniform{n}"]), 3)
               for n in UNIFORM_NS},
        },
        "delta_cider": {
            "mean": round(statistics.mean(delta_qs), 3),
            "median": round(statistics.median(delta_qs), 3),
            "std": round(statistics.pstdev(delta_qs), 3),
        },
        "location_sensitivity": {
            "mean_per_video_std": round(statistics.mean(per_video_loc_std), 3),
            "median_per_video_std": round(statistics.median(per_video_loc_std), 3),
            "max_per_video_std": round(max(per_video_loc_std), 3),
        },
        "caption_diversity": {
            "mean_unique_captions_per_video": round(statistics.mean(unique_per_video), 3),
        },
        "positive_delta_ratio": round(
            sum(1 for d in delta_qs if d > 0) / max(len(delta_qs), 1), 3),
    }
    return stats


def eval_val_metrics(captioner, val_images, val_refs, val_features_dir, device=None):
    """Captioning metrics on the FIXED val set (74 videos, cached features)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_features_dir = Path(val_features_dir)
    preds = {}
    with torch.no_grad():
        for img in val_images:
            f_path = val_features_dir / f"{img['id']}.pt"
            assert f_path.exists(), f"missing val features for {img['id']}"
            feats = torch.load(f_path, map_location=device).unsqueeze(0)
            mask = torch.ones(1, feats.shape[1], dtype=torch.long, device=device)
            cap = generate_caption(captioner, visual_features=feats, frame_mask=mask)[0]
            preds[str(img["id"])] = [{"sentence": cap}]
    import sys
    from pathlib import Path as _P
    cocap_root = _P("third_party/CoCap").resolve()
    if str(cocap_root) not in sys.path:
        sys.path.insert(0, str(cocap_root))
    from cocap.modeling.eval_captioning import evaluate
    refs_sub = {str(k): v for k, v in val_refs.items() if str(k) in preds}
    metrics = {k: round(v * 100, 2) for k, v in
               evaluate(submission={"results": preds}, reference=refs_sub).items()
               if isinstance(v, float)}
    return metrics
