#!/usr/bin/env python
"""Verify ERA_Dataset.zip against the official CapERA annotation JSONs.

Read-only: checks ZIP integrity, size, entry list, and matches archive
video files against the video_ids in datasets/CapERA/annotations/*.json.
Does NOT extract, convert, or modify anything.

Matching is done per split: CapERA_DATASET_train.json <-> Videos/Tra,
CapERA_DATASET_test.json <-> Videos/Test. This is required because the
video_id strings overlap across splits (1391 test id strings also appear
in train) — they name DIFFERENT videos in different folders, so matching
by video_id alone undercounts. Archive filenames may also contain a
space before the extension (e.g. "Baseball_001 .mp4"), which is
normalized before comparison.

Usage: python scripts/verify_capera_era.py [path-to-ERA_Dataset.zip]
"""
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "datasets/CapERA/videos/ERA_Dataset.zip"
ANN_DIR = ROOT / "datasets/CapERA/annotations"

SPLIT_MAP = {"train": "Tra", "test": "Test"}  # annotation JSON -> archive folder


def norm_ext(name):
    """Strip whitespace right before the extension: 'X .mp4' -> 'X.mp4'."""
    return re.sub(r"\s+\.(\w+)$", r".\1", name)


def vid_class(video_id):
    """'Baseball_001.mp4' -> 'Baseball'."""
    return video_id.rsplit("_", 1)[0]


# --- 1. collect expected video ids from the official CapERA JSONs, per split ---
expected = {}
for split, json_name in [("train", "CapERA_DATASET_train.json"),
                         ("test", "CapERA_DATASET_test.json")]:
    d = json.load(open(ANN_DIR / json_name))
    ids = [norm_ext(item["video_id"]) for item in d["ERA_caption"]]
    # key by (class, id) so a file in the wrong class dir is caught
    expected[split] = {(vid_class(i), i) for i in ids}
    print(f"[annotations] {split}: {len(expected[split])} unique video ids")

overlap = len({i for _, i in expected["train"]} & {i for _, i in expected["test"]})
print(f"[annotations] id-string overlap across splits: {overlap} "
      f"(expected — ids repeat per split, matching must be per split)")

# --- 2. file size ---
size = ZIP_PATH.stat().st_size
print(f"[file] {ZIP_PATH} = {size / 1e9:.2f} GB decimal "
      f"(~{size / 2**30:.2f} GiB, expect ~6.29 GiB)")

# --- 3. zip integrity + listing (no extraction) ---
zf = zipfile.ZipFile(ZIP_PATH)
bad = zf.testzip()
if bad is not None:
    print(f"[integrity] FAIL — corrupt entry: {bad}")
    sys.exit(1)
print("[integrity] OK — all entries readable")

entries = zf.namelist()
archive = {split: set() for split in SPLIT_MAP}
for e in entries:
    parts = e.split("/")
    if len(parts) != 4 or not parts[3].lower().endswith(".mp4"):
        continue
    if parts[0] == "Videos" and parts[1] in ("Tra", "Test"):
        split = "train" if parts[1] == "Tra" else "test"
        archive[split].add((parts[2], norm_ext(parts[3])))

n_mp4 = sum(len(v) for v in archive.values())
print(f"[contents] {len(entries)} entries, {n_mp4} .mp4 files "
      f"(Tra: {len(archive['train'])}, Test: {len(archive['test'])})")

# --- 4. match against annotations, per split ---
total_found = total_expected = 0
ok = True
for split in ("train", "test"):
    folder = SPLIT_MAP[split]
    exp, got = expected[split], archive[split]
    found = exp & got
    missing = exp - got
    extra = got - exp
    total_found += len(found)
    total_expected += len(exp)
    print(f"[match] {split} (Videos/{folder}): {len(found)}/{len(exp)} "
          f"({100 * len(found) / len(exp):.1f}%)")
    if missing:
        ok = False
        print(f"[match] {split} MISSING {len(missing)}, e.g.: {sorted(missing)[:10]}")
    if extra:
        ok = False
        print(f"[match] {split} extra files not in annotations: {len(extra)}, "
              f"e.g.: {sorted(extra)[:10]}")

print(f"[match] total: {total_found}/{total_expected} "
      f"({100 * total_found / total_expected:.1f}%)")
sys.exit(0 if ok and total_found == total_expected else 1)
