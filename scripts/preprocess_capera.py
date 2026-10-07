"""CapERA annotation preprocess → xmodaler protocol.

Input (read-only, never modified):
    datasets/CapERA/annotations/CapERA_DATASET_{train,test}.json
    datasets/CapERA/videos/Videos/{Tra,Test}/  (only used to validate id <-> file matching)

Output (all new artifacts):
    data/CapERA/video_id_map.json            int id <-> original video_id (global unique ints)
    data/CapERA/vocabulary.txt               1 word per line; id 0 reserved for <BOS>/<EOS>
    data/CapERA/capera_caption_anno_{train,val,test}.pkl
                                             [{'video_id': int, 'tokens_ids': (1,W) uint32, 'target_ids': (1,W) int32}]
                                             one entry per caption; W = MAX_SEQ_LEN
    data/CapERA/captions_{val,test}.json     COCO-style for COCOEvaler (5 refs per video)

Token protocol (matches xmodaler tools/msvd_preprocess.py):
    tokens_ids = [0(BOS), w1, ..., wn, 0(pad)]
    target_ids = [w1, ..., wn, 0(EOS), -1(pad)]
    vocabulary.txt line i -> id i+1
"""
import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "CapERA")
ANNO_DIR = os.path.join(PROJECT_ROOT, "datasets", "CapERA", "annotations")
VIDEOS_ROOT = os.path.join(PROJECT_ROOT, "datasets", "CapERA", "videos", "Videos")

SPLIT_DIRS = {"train": "Tra", "test": "Test"}


def load_capera(split):
    path = os.path.join(ANNO_DIR, f"CapERA_DATASET_{split}.json")
    with open(path, "r") as f:
        data = json.load(f)
    items = data["ERA_caption"]
    out = []
    for e in items:
        vid = e["video_id"]
        caps = e["annotation"]["English_caption"]
        assert isinstance(caps, list) and len(caps) == 5, (vid, len(caps))
        out.append((vid, [c.strip() for c in caps]))
    return out


def build_video_file_map():
    """video_id (string) -> absolute mp4 path, for both splits."""
    fmap = {}
    for split, sub in SPLIT_DIRS.items():
        root = os.path.join(VIDEOS_ROOT, sub)
        for cat in sorted(os.listdir(root)):
            cat_dir = os.path.join(root, cat)
            if not os.path.isdir(cat_dir):
                continue
            for fn in os.listdir(cat_dir):
                if fn.endswith(".mp4"):
                    fmap[(split, fn)] = os.path.join(cat_dir, fn)
    return fmap


def tokenize_captions(caption_lists):
    """PTB tokenization (same tokenizer family as pycocoevalcap evaluation)."""
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    ann = {}
    idx = 0
    order = []  # (video_id, caption_index) -> tokenizer ann id
    for vid, caps in caption_lists:
        for ci, c in enumerate(caps):
            ann[idx] = [{"caption": c}]
            order.append((vid, ci))
            idx += 1
    tok = PTBTokenizer().tokenize(ann)
    tokens_by_vid = {}
    for i, (vid, ci) in enumerate(order):
        toks = tok[i][0].split()  # PTB returns list with one token-string per caption
        tokens_by_vid.setdefault(vid, []).append(toks)
    return tokens_by_vid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DATA_DIR)
    parser.add_argument("--val-size", type=int, default=74, help="number of train videos held out as val")
    parser.add_argument("--val-seed", type=int, default=42)
    parser.add_argument("--max-caption-len", type=int, default=30, help="truncate captions to this many tokens")
    parser.add_argument("--word-count-threshold", type=int, default=0,
                        help="words with count <= threshold map to UNK (0 = keep all words)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load + verify videos exist ----
    train_items = load_capera("train")   # [(video_id, [5 captions])]
    test_items = load_capera("test")
    fmap = build_video_file_map()
    for split, items in (("train", train_items), ("test", test_items)):
        missing = [vid for vid, _ in items if (split, vid) not in fmap]
        assert not missing, f"[{split}] videos missing on disk: {missing[:10]}"
        assert len({vid for vid, _ in items}) == len(items), f"[{split}] duplicate video ids"
    print(f"verified files on disk: train={len(train_items)} test={len(test_items)}")

    # ---- global unique int ids: train 0..N-1, test N.. ----
    train_items = sorted(train_items, key=lambda x: x[0])
    test_items = sorted(test_items, key=lambda x: x[0])
    id_map = {"train": {}, "test": {}}
    id_to_meta = {}
    cursor = 0
    for split, items in (("train", train_items), ("test", test_items)):
        for vid, _ in items:
            id_map[split][vid] = cursor
            id_to_meta[cursor] = {"video_id": vid, "split": split,
                                  "path": fmap[(split, vid)]}
            cursor += 1
    with open(os.path.join(args.out_dir, "video_id_map.json"), "w") as f:
        json.dump({"ids": {str(k): v for k, v in id_to_meta.items()},
                   "by_split": id_map}, f, indent=1)
    print(f"int ids assigned: train 0..{len(train_items)-1}, test {len(train_items)}..{cursor-1}")

    # ---- tokenize all captions (PTB); key by (split, video_id): id strings
    # overlap across splits, so string keys alone would collide ----
    tokens_by_vid = {}
    tokens_by_vid.update({("train", v): c for v, c in tokenize_captions(train_items).items()})
    tokens_by_vid.update({("test", v): c for v, c in tokenize_captions(test_items).items()})

    lens = [len(t) for caps in tokens_by_vid.values() for t in caps]
    print(f"tokenized caption lengths: min={min(lens)} max={max(lens)} "
          f"mean={np.mean(lens):.1f} p99={int(np.percentile(lens, 99))}")
    max_len = min(args.max_caption_len, int(np.percentile(lens, 99)))
    print(f"max caption tokens (truncated): {max_len} -> MAX_SEQ_LEN={max_len + 2}")

    # ---- vocab ----
    counts = {}
    for caps in tokens_by_vid.values():
        for toks in caps:
            for w in toks:
                counts[w] = counts.get(w, 0) + 1
    thr = args.word_count_threshold
    vocab = sorted([w for w, n in counts.items() if n > thr],
                   key=lambda w: (-counts[w], w))
    if any(n <= thr for n in counts.values()):
        vocab.append("UNK")
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}
    with open(os.path.join(args.out_dir, "vocabulary.txt"), "w") as f:
        f.write("\n".join(vocab) + "\n")
    print(f"vocab size: {len(vocab)} (VOCAB_SIZE={len(vocab) + 1}, includes id 0)")

    # ---- encode into pkl (per-caption entries, xmodaler protocol) ----
    width = max_len + 2  # BOS + up to max_len words + EOS
    rng = np.random.default_rng(args.val_seed)
    val_ids = set(rng.choice(len(train_items), size=args.val_size, replace=False))

    annos = {"train": [], "val": [], "test": []}
    coco = {s: {"images": [], "annotations": []} for s in ("val", "test")}
    ann_cursor = 0

    def encode(vid, toks, split, int_id, raw_split):
        nonlocal ann_cursor
        toks = toks[:max_len]
        input_Li = np.zeros((1, width), dtype="uint32")
        output_Li = np.zeros((1, width), dtype="int32") - 1
        unk = wtoi.get("UNK", None)
        for k, w in enumerate(toks):
            wid = wtoi.get(w, unk)
            if wid is None:  # cannot happen with threshold 0, defensive
                continue
            input_Li[0, k + 1] = wid
            output_Li[0, k] = wid
        output_Li[0, len(toks)] = 0  # EOS
        annos[split].append({"video_id": int_id,
                             "tokens_ids": input_Li,
                             "target_ids": output_Li})
        if split != "train":
            coco[split]["images"].append({"id": int_id, "file_name": vid})
            for raw in raw_caps[(raw_split, vid)]:
                coco[split]["annotations"].append(
                    {"image_id": int_id, "id": ann_cursor, "caption": raw})
                ann_cursor += 1

    raw_caps = {}
    for split, items in (("train", train_items), ("test", test_items)):
        for vid, caps in items:
            raw_caps[(split, vid)] = caps

    for i, (vid, caps) in enumerate(train_items):
        int_id = id_map["train"][vid]
        if i in val_ids:
            # val/test pkl: ONE entry per video (generation mode needs no
            # captions; duplicate video_ids would yield 5 predictions per video
            # and break pycocoevalcap's len(hypo)==1 assertion)
            encode(vid, tokens_by_vid[("train", vid)][0], "val", int_id, raw_split="train")
        else:
            for toks in tokens_by_vid[("train", vid)]:
                encode(vid, toks, "train", int_id, raw_split="train")

    for vid, caps in test_items:
        int_id = id_map["test"][vid]
        encode(vid, tokens_by_vid[("test", vid)][0], "test", int_id, raw_split="test")

    import pickle
    for split in ("train", "val", "test"):
        with open(os.path.join(args.out_dir, f"capera_caption_anno_{split}.pkl"), "wb") as f:
            pickle.dump(annos[split], f)
        print(f"{split}: {len(annos[split])} caption entries")
    for split in ("val", "test"):
        with open(os.path.join(args.out_dir, f"captions_{split}.json"), "w") as f:
            json.dump(coco[split], f)
        n_img = len(coco[split]["images"])
        n_ann = len(coco[split]["annotations"])
        assert n_ann == 5 * n_img, (split, n_img, n_ann)
        print(f"captions_{split}.json: {n_img} images, {n_ann} annotations (5/video)")

    print("done. outputs in", args.out_dir)


if __name__ == "__main__":
    main()
