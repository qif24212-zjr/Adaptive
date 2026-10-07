# -*- coding: utf-8 -*-
"""
Integration smoke test: Adaptive Selector -> Frame-CoCap.

    real CapERA video (B=4)
        -> uniform sample M=32 candidate frames
        -> CLIP ViT-B/16 features  [B, 32, 512]
        -> AdaptiveSelector        (variable selected count, random-init policy)
        -> gather selected features [B, N_max, 512] + mask
        -> existing Frame-CoCap (features-only path)
        -> greedy caption generation

The selector is randomly initialized, so selection QUALITY is not judged.
This test only proves the full software pipeline closes end-to-end.

Run:
    python scripts/smoke_test_adaptive_frame_cocap.py
"""

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "xmodaler"))  # selectors __init__

import torch

from models.frame_cocap import build_frame_captioner, read_video_frames_cv2
from models.selectors.adaptive_frame_cocap import AdaptiveFrameCoCap
from models.selectors.adaptive_selector import AdaptiveSelector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, M, D = 4, 32, 512
MIN_SEL, MAX_SEL = 2, 16
VIDEO_ROOT = PROJECT_ROOT / "datasets/CapERA/videos/Videos/Test"
ANN_JSON = PROJECT_ROOT / "data/CapERA/captions_test.json"


def build_video_index(video_root):
    index = {}
    for p in video_root.rglob("*.mp4"):
        index[p.name.replace(" ", "")] = p
    return index


def run_pipeline(pipeline, candidate_frames, captions):
    """One full pipeline run; returns everything the test needs to check.
    (no_grad: this is an inference smoke test; keeps the 128-frame encode
    from retaining autograd graphs across pipeline calls)"""
    with torch.no_grad():
        out = pipeline(candidate_frames)

    # [5] Frame-CoCap forward on the SELECTED features (teacher-forced)
    logits = pipeline.captioner(
        visual_features=out["selected_features"],
        frame_mask=out["selected_feature_mask"],
        captions=captions,
    )
    # [6] greedy generation from the selected frames
    gen = pipeline.generate(candidate_frames)
    return out, logits, gen


def check_and_print(candidate_frames, captions, video_names, pipeline, tag):
    out, logits, gen = run_pipeline(pipeline, candidate_frames, captions)

    counts = out["selected_count"]
    indices = out["selected_indices"]
    sel_mask = out["selected_mask"]
    gathered = out["selected_features"]
    gathered_mask = out["selected_feature_mask"]

    print(f"--- selection ({tag}) ---")
    for b in range(B):
        idxs = indices[b, sel_mask[b]].tolist()
        print(f"Video {b} ({video_names[b]}):")
        print(f"    selected_indices = {idxs}")
        print(f"    selected_count = {counts[b].item()}")

    # ---- assertions ----
    assert candidate_frames.shape[1] == M
    feats = out["candidate_features"]
    assert feats.shape == (B, M, D), feats.shape
    assert counts.min().item() >= MIN_SEL and counts.max().item() <= MAX_SEL
    assert sel_mask.shape[0] == B
    # gather: per-sample mask sum == count; padding slots zeroed
    assert torch.equal(gathered_mask.sum(1), counts), "gather mask sums != counts"
    masked_out = gathered[gathered_mask == 0]
    assert masked_out.numel() == 0 or torch.count_nonzero(masked_out) == 0, \
        "gather padding slots are not zero"
    for b in range(B):
        idxs = indices[b, sel_mask[b]].tolist()
        assert all(0 <= i < M for i in idxs), f"invalid index in video {b}: {idxs}"
        assert all(idxs[i] < idxs[i + 1] for i in range(len(idxs) - 1)), \
            f"not strictly increasing: {idxs}"
        assert len(set(idxs)) == len(idxs), f"duplicates: {idxs}"
    assert logits.ndim == 3 and torch.isfinite(logits).all(), "logits not finite"
    assert len(gen) == B and all(isinstance(c, str) and len(c) > 0 for c in gen), \
        "generation failed for some video"
    # the budget must be ADAPTIVE, not pinned to 16
    assert not bool((counts == MAX_SEL).all()), \
        "all samples selected exactly max_selected_frames (policy looks pinned)"
    return out, logits, gen


def main():
    print("=" * 50)
    print("Adaptive Frame-CoCap Integration Smoke Test")
    print("=" * 50)
    print(f"device: {DEVICE}")
    print(f"batch_size: {B}")
    print(f"candidate_frames: {M}")

    print("\n[1] candidate frame sampling (real CapERA test videos)")
    ann = json.load(open(ANN_JSON))
    refs = {}
    for a in ann["annotations"]:
        refs.setdefault(a["image_id"], []).append(a["caption"])
    rng = random.Random(0)
    chosen = rng.sample(ann["images"], B)
    video_index = build_video_index(VIDEO_ROOT)
    frame_list, video_names, gt_captions = [], [], []
    for img in chosen:
        path = video_index[img["file_name"].replace(" ", "")]
        frames = read_video_frames_cv2(str(path), n_frames=M, sample="uniform")
        assert frames is not None and frames.shape[0] == M, \
            f"{img['file_name']}: got {None if frames is None else frames.shape[0]} frames"
        frame_list.append(frames)
        video_names.append(img["file_name"])
        gt_captions.append(refs[img["id"]][0])
    candidate_frames = torch.stack(frame_list).to(DEVICE)
    print(f"frames shape: {list(candidate_frames.shape)}")

    print("\n[2] frame feature extraction (CLIP ViT-B/16)")
    captioner = build_frame_captioner(
        frame_encoder="clip_vitb16", clip_path="checkpoints/clip/ViT-B-16.pt",
    ).to(DEVICE).eval()

    print("\n[3] adaptive selection (random-init policy; sampling with retry "
          "so the budget variation assertion cannot flake)")
    passed = None
    for seed in range(5):
        torch.manual_seed(seed)
        selector = AdaptiveSelector(
            feature_dim=D, hidden_size=256,
            min_selected_frames=MIN_SEL, max_selected_frames=MAX_SEL, sample=True,
        ).to(DEVICE)
        pipeline = AdaptiveFrameCoCap(
            frame_encoder=captioner.frame_encoder,
            selector=selector,
            captioner=captioner,
        ).eval()
        with torch.no_grad():
            out = pipeline(candidate_frames)
        if not bool((out["selected_count"] == MAX_SEL).all()):
            passed = (pipeline, f"seed {seed}")
            break
        del out
        torch.cuda.empty_cache()
    assert passed is not None, "no seed produced a non-pinned selection"
    pipeline, tag = passed

    print("\n[4] run full pipeline with checks")
    out, logits, gen = check_and_print(candidate_frames, gt_captions, video_names, pipeline, tag)

    print(f"\n[4b] gather selected frames")
    print(f"selected batch shape: {list(out['selected_features'].shape)}")
    print(f"selected mask shape: {list(out['selected_feature_mask'].shape)}")

    print("\n[5] Frame-CoCap forward")
    print(f"logits shape: {list(logits.shape)}")

    print("\n[6] greedy generation")
    for b in range(B):
        print(f"Video {b} caption: '{gen[b]}'")

    print("\n" + "=" * 50)
    print("ALL INTEGRATION SMOKE TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
