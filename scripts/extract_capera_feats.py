"""CapERA video -> per-frame visual features (offline, xmodaler protocol).

For each video: uniformly sample T = round(5s * fps) frames, extract per-frame
features with a frozen backbone, save as npz {'features': (T, D) float32} at
    features/CapERA/{backbone}/{split}/{int_id}.npz

Backbones:
    maxvit_s   timm 'maxvit_small_tf_224'  -> 768-d   (primary, CapERA-paper backbone)
    resnet152  timm 'resnet152'            -> 2048-d  (xmodaler MSVD protocol dim, ablation)

Reads (read-only): datasets/CapERA/videos/... , data/CapERA/video_id_map.json
Writes: features/CapERA/... (new artifacts only)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import torch
import timm
from PIL import Image
from tqdm import tqdm


def build_model(backbone, device):
    model = timm.create_model(backbone, pretrained=True, num_classes=0)
    model = model.to(device).eval()
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)
    return model, transform


def sample_frames(cap, fps):
    """Uniformly sample frames from a 5s video at `fps`."""
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = max(1, round(5 * fps))
    indices = [int((i + 0.5) * n_total / target) for i in range(target)]
    indices = sorted(set(i for i in indices if 0 <= i < n_total))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="maxvit_small_tf_224",
                        choices=["maxvit_small_tf_224", "resnet152"])
    parser.add_argument("--fps", type=float, default=2.0,
                        help="candidate frames per second (2 -> 10 frames per 5s video)")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--id-map", default=os.path.join(PROJECT_ROOT, "data", "CapERA", "video_id_map.json"))
    parser.add_argument("--out-dir", default=None,
                        help="default: features/CapERA/{backbone}")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="only process the first N videos per split (for smoke tests)")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, backbone: {args.backbone}, fps: {args.fps}")

    if args.out_dir is None:
        args.out_dir = os.path.join(PROJECT_ROOT, "features", "CapERA", args.backbone)

    with open(args.id_map) as f:
        id_map = json.load(f)
    meta_by_int = {int(k): v for k, v in id_map["ids"].items()}

    model, transform = build_model(args.backbone, device)

    # videos to process, ordered by int id
    tasks = []
    for int_id, meta in sorted(meta_by_int.items(), key=lambda kv: kv[0]):
        if meta["split"] in args.splits:
            tasks.append((int_id, meta))
    if args.max_videos is not None:
        # keep first N per split (ids are split-contiguous: train first, then test)
        kept, seen = [], {}
        for int_id, meta in tasks:
            seen[meta["split"]] = seen.get(meta["split"], 0) + 1
            if seen[meta["split"]] <= args.max_videos:
                kept.append((int_id, meta))
        tasks = kept
    print(f"videos to process: {len(tasks)}")

    t_start = time.time()
    n_done, n_skip, n_fail = 0, 0, 0
    shape_min, shape_max = None, None

    for int_id, meta in tqdm(tasks):
        split = meta["split"]
        out_path = os.path.join(args.out_dir, split, f"{int_id}.npz")
        if args.skip_existing and os.path.exists(out_path):
            n_skip += 1
            continue

        cap = cv2.VideoCapture(meta["path"])
        frames = sample_frames(cap, args.fps)
        cap.release()
        if len(frames) == 0:
            print(f"  [warn] no frames decoded: {meta['video_id']}")
            n_fail += 1
            continue

        imgs = [transform(Image.fromarray(img)) for img in frames]
        imgs = torch.stack(imgs).to(device)

        feats = []
        with torch.no_grad():
            for i in range(0, imgs.size(0), args.batch_size):
                batch = imgs[i:i + args.batch_size]
                feats.append(model(batch).float().cpu())
        feats = torch.cat(feats, 0).numpy()  # (T, D)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, features=feats.astype("float32"))
        n_done += 1
        if shape_min is None or feats.shape[0] < shape_min[0]:
            shape_min = feats.shape
        if shape_max is None or feats.shape[0] > shape_max[0]:
            shape_max = feats.shape

    dt = time.time() - t_start
    print(f"done: {n_done} extracted, {n_skip} skipped, {n_fail} failed, {dt:.1f}s")
    if shape_min is not None:
        print(f"feature shape range (T, D): min={tuple(shape_min)} max={tuple(shape_max)}")
    # sanity: read one file back
    if n_done > 0:
        one = next(f for f in os.listdir(os.path.join(args.out_dir, tasks[0][1]["split"])) if f.endswith(".npz"))
        probe = np.load(os.path.join(args.out_dir, tasks[0][1]["split"], one))
        print(f"read-back check {one}: features {probe['features'].shape} "
              f"dtype {probe['features'].dtype}, NaN: {bool(np.isnan(probe['features']).any())}")


if __name__ == "__main__":
    main()
