#!/usr/bin/env python
"""MSVD unified data-interface checker.

Verifies the stability of the four-way contract used by all downstream
experiments:

    video_id  ->  video file  ->  caption  ->  split

Exit code 0 = interface OK, 1 = broken.

Usage: python check_msvd.py [--data-root DIR]
"""
import argparse
import json
import os
import sys

DEFAULT_ROOT = "/root/autodl-tmp/uav_adaptive_captioning/datasets/MSVD"
EXPECTED = {"videos": 1970, "train": 1200, "val": 100, "test": 670}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_ROOT)
    args = ap.parse_args()
    root = args.data_root

    problems = 0

    captions = json.load(open(os.path.join(root, "captions", "MSVD_caption.json"),
                              encoding="utf-8"))
    mapping = json.load(open(os.path.join(root, "captions",
                                          "video_name_mapping.json"),
                             encoding="utf-8"))
    splits, split_of = {}, {}
    for name in ("train", "val", "test"):
        with open(os.path.join(root, "splits", f"{name}_list.txt")) as fh:
            ids = [l.strip() for l in fh if l.strip()]
        splits[name] = ids
        for vid in ids:
            split_of[vid] = name

    print(f"captions : {len(captions)} videos "
          f"({sum(len(v) for v in captions.values())} captions)")
    print(f"splits   : " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

    if len(captions) != EXPECTED["videos"]:
        print(f"FAIL: expected {EXPECTED['videos']} caption videos"); problems += 1
    for name, exp in EXPECTED.items():
        if name != "videos" and len(splits[name]) != exp:
            print(f"FAIL: split {name} = {len(splits[name])}, expected {exp}")
            problems += 1

    union = set().union(*(set(v) for v in splits.values()))
    if len(union) != EXPECTED["videos"] or any(
            len(set(splits[a]) & set(splits[b]))
            for i, a in enumerate(splits) for b in list(splits)[i + 1:]):
        print("FAIL: splits do not partition 1970 videos"); problems += 1
    if set(captions) != union:
        print("FAIL: caption ids != split union"); problems += 1

    missing_file, missing_map, missing_cap, missing_split = [], [], [], []
    for vid in captions:
        if vid not in mapping:
            missing_map.append(vid)
        elif not os.path.isfile(os.path.join(root, "videos", mapping[vid])):
            missing_file.append(vid)
        if vid not in split_of:
            missing_split.append(vid)
    for f in os.listdir(os.path.join(root, "videos")):
        if os.path.splitext(f)[0] not in captions:
            missing_cap.append(f)
    print(f"video_id->file broken : {len(missing_file)}")
    print(f"video_id->split broken: {len(missing_split)}")
    print(f"file->caption broken  : {len(missing_cap)}")
    problems += bool(missing_file or missing_map or missing_cap or missing_split)

    print("\nINTERFACE " + ("OK" if problems == 0 else "BROKEN"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
