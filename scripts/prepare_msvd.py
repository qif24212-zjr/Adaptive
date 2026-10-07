#!/usr/bin/env python
"""MSVD unified data-interface preparation.

Pipeline contract (shared by ALL downstream methods):
    video_id  ->  video file  ->  caption  ->  split

Every method must consume the SAME 32-frame candidate pool:
    32 uniformly sampled candidate frames -> candidate features
    -> 8-frame selection -> captioning model.

This script only builds the manifest and the candidate-frame sampling API.
It does NOT extract visual features.

Usage:
    python prepare_msvd.py [--data-root DIR] [--out manifest.json]
    python prepare_msvd.py --extract-frames --frames-out DIR   # optional JPEG dump
"""
import argparse
import json
import os
import sys

DEFAULT_ROOT = "/root/autodl-tmp/uav_adaptive_captioning/datasets/MSVD"
SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def uniform_frame_indices(frame_count, n=32):
    """Uniformly sample `n` candidate frame indices from a video.

    Returns up to `n` distinct 0-based indices (fewer if frame_count < n).
    """
    frame_count = int(frame_count)
    if frame_count <= 0:
        return []
    n = min(int(n), frame_count)
    idx = [round((frame_count - 1) * i / (n - 1)) for i in range(n)] if n > 1 else [0]
    # dedupe while preserving order
    return list(dict.fromkeys(idx))


def load_msvd(data_root=DEFAULT_ROOT):
    """Load the standard MSVD files. Returns (captions, mapping, splits)."""
    captions = json.load(open(os.path.join(data_root, "captions",
                                           "MSVD_caption.json"), encoding="utf-8"))
    mapping = json.load(open(os.path.join(data_root, "captions",
                                          "video_name_mapping.json"), encoding="utf-8"))
    splits = {}
    for name in ("train", "val", "test"):
        with open(os.path.join(data_root, "splits", f"{name}_list.txt")) as fh:
            splits[name] = [l.strip() for l in fh if l.strip()]
    return captions, mapping, splits


def build_manifest(data_root=DEFAULT_ROOT, out_path=None):
    """Write a stable manifest: video_id -> file/caption/split, one JSON."""
    captions, mapping, splits = load_msvd(data_root)
    split_of = {}
    for name, ids in splits.items():
        for vid in ids:
            split_of[vid] = name
    records = []
    for vid in sorted(captions, key=lambda v: (SPLIT_ORDER[split_of[v]], v)):
        records.append({
            "video_id": vid,
            "filename": mapping[vid],
            "path": os.path.join("videos", mapping[vid]),
            "split": split_of[vid],
            "captions": captions[vid],
        })
    out = out_path or os.path.join(data_root, "msvd_manifest.json")
    json.dump(records, open(out, "w"), ensure_ascii=False, indent=1)
    return records


def extract_candidate_frames(data_root=DEFAULT_ROOT, frames_dir=None,
                             n_candidates=32):
    """Optional: dump the 32-frame candidate pool as JPEGs.

    NOT part of the default pipeline; features are never extracted here.
    """
    import cv2
    captions, mapping, splits = load_msvd(data_root)
    frames_dir = frames_dir or os.path.join(data_root, "candidate_frames")
    os.makedirs(frames_dir, exist_ok=True)
    for vid in sorted(captions):
        path = os.path.join(data_root, "videos", mapping[vid])
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = uniform_frame_indices(total, n_candidates)
        vid_dir = os.path.join(frames_dir, vid)
        os.makedirs(vid_dir, exist_ok=True)
        for i, fi in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(os.path.join(vid_dir, f"{i:02d}_{fi:05d}.jpg"), frame)
        cap.release()
    return frames_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=None, help="manifest output path")
    ap.add_argument("--extract-frames", action="store_true",
                    help="also dump 32-frame candidate pool as JPEGs")
    ap.add_argument("--frames-out", default=None)
    ap.add_argument("--candidate-frames", type=int, default=32)
    args = ap.parse_args()

    records = build_manifest(args.data_root, args.out)
    n = {s: sum(1 for r in records if r["split"] == s) for s in SPLIT_ORDER}
    print(f"manifest: {len(records)} videos "
          f"(train {n['train']} / val {n['val']} / test {n['test']})")
    print(f"written: {args.out or os.path.join(args.data_root, 'msvd_manifest.json')}")

    if args.extract_frames:
        d = extract_candidate_frames(args.data_root, args.frames_out,
                                     args.candidate_frames)
        print(f"candidate frames dumped to {d}")


if __name__ == "__main__":
    sys.exit(main())
