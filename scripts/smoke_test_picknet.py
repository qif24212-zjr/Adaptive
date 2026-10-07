"""PickNet-style selector skeleton smoke tests (Phase 2 gate).

Test 1: dataset candidates (B,10,768) + selector output shapes (2-4 videos)
Test 2: per-video selection traces; variable counts, first frame forced PICK
Test 3: full XE forward through SelectorTransformerEncoderDecoder
Test 4: backward (XE + budget regularizer), selector gradients flow
Test 5: 4-video overfit (200 steps), loss decreases
Test 6: inference + beam search on 2 videos
Test 7: COCO evaluation chain on the full val split (74 videos) + trace dump

Usage: python scripts/smoke_test_picknet.py [--test 1|2|3|4|5|6|7|all]
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XMODALER_DIR = os.path.join(PROJECT_ROOT, "third_party", "xmodaler")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, XMODALER_DIR)

import xmodaler.datasets.videos.capera  # noqa: F401
import models.selector_enc_dec  # noqa: F401  (meta arch + SELECTOR schema + selectors)

import pycocoevalcap.eval as _coco_eval


class _DummySpice:
    def __init__(self):
        pass

    def method(self):
        return "SPICE"

    def compute_score(self, *args, **kwargs):
        return 0.0, []


_coco_eval.Spice = _DummySpice

import numpy as np
import torch
import tqdm

from xmodaler.config import get_cfg, kfg
from xmodaler.datasets import (build_dataset_mapper,
                               build_xmodaler_valtest_loader)
from xmodaler.evaluation import build_evaluation
from xmodaler.losses import build_losses
from xmodaler.modeling import add_config, build_model
from xmodaler.optim import build_optimizer
from xmodaler.utils.env import seed_all_rng

from models.selectors.base_selector import (SEL_INDICES, SEL_NUM_CANDIDATES,
                                            SEL_NUM_SELECTED, SEL_PROBS)
from models.selectors.picknet_style_selector import PickNetStyleSelector

CONFIG = os.path.join(PROJECT_ROOT, "configs", "capera", "picknet_style.yaml")


def build_cfg(base_lr=None):
    cfg = get_cfg()
    tmp_cfg = cfg.load_from_file_tmp(CONFIG)
    add_config(cfg, tmp_cfg)
    cfg.merge_from_file(CONFIG)
    if base_lr is not None:
        cfg.defrost()
        cfg.SOLVER.BASE_LR = base_lr
        cfg.freeze()
    seed_all_rng(42)
    return cfg


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  [OK] {msg}")


def get_samples(cfg, n_train=4, n_test=2):
    mapper_train = build_dataset_mapper(cfg, name=cfg.DATASETS.TRAIN, stage="train")
    train_list = mapper_train.load_data(cfg)
    train_samples = [mapper_train(train_list[i]) for i in range(n_train)]
    mapper_test = build_dataset_mapper(cfg, name=cfg.DATASETS.TEST, stage="test")
    test_list = mapper_test.load_data(cfg)
    test_samples = [mapper_test(test_list[i]) for i in range(n_test)]
    return train_samples, test_samples, mapper_train, train_list


def test1_candidates_and_selector(cfg):
    print("=== Test 1: dataset candidates (B,10,768) + selector shapes ===")
    train_samples, test_samples, _, _ = get_samples(cfg, n_train=4, n_test=2)

    # dataset must provide ALL candidates, not uniform-8
    for s in train_samples:
        check(s["ATT_FEATS"].shape == (10, 768),
              f"dataset ATT_FEATS shape {tuple(s['ATT_FEATS'].shape)} == (10, 768)")

    model = build_model(cfg).cuda().eval()
    data = model.preprocess_batch(train_samples)  # selector runs inside
    check(data["ATT_FEATS"].shape == (4, 10, 768),
          f"selected ATT_FEATS shape {tuple(data['ATT_FEATS'].shape)} == (4, 10, 768)")
    mask = data["ATT_MASKS"]
    check(mask.shape == (4, 10) and mask[:, 0].sum() == 4,
          "selection mask (4,10) with first frame forced selected")
    check(torch.allclose(mask.sum(1), data[SEL_NUM_SELECTED].float()),
          "mask row sums == NUM_SELECTED")
    dropped_ok = all(torch.allclose(data["ATT_FEATS"][b][mask[b] == 0],
                                    torch.zeros(768, device=data["ATT_FEATS"].device))
                     for b in range(4))
    check(dropped_ok, "dropped positions are zero-padded")
    check(tuple(data[SEL_NUM_CANDIDATES].tolist()) == (10, 10, 10, 10),
          "NUM_CANDIDATES == 10 per video")
    return model, train_samples, test_samples


def test2_selection_traces(cfg, train_samples):
    print("=== Test 2: per-video selection traces (variable-length proof) ===")
    # standalone selector with neutral init prob to show decision variability
    selector = PickNetStyleSelector(input_dim=768, hidden_size=256,
                                    init_pick_prob=0.5).cuda()
    model = build_model(cfg).cuda()
    model.selector = selector  # swap in for this test

    data = model.preprocess_batch(train_samples[:4])
    with torch.no_grad():
        model.eval()
        data_eval = model.preprocess_batch(train_samples[:4])

    for b in range(4):
        idxs = data_eval[SEL_INDICES][b]
        probs = data_eval[SEL_PROBS][b].tolist()
        print(f"    video {b}: candidates = 10, selected = {idxs}, "
              f"num_selected = {len(idxs)}")
        check(0 in idxs, f"video {b}: frame 0 is selected")
        check(1 <= len(idxs) <= 10, f"video {b}: num_selected in [1,10]")

    counts = {len(data_eval[SEL_INDICES][b]) for b in range(4)}
    # stochastic training-mode draws: demonstrate the mechanism is per-frame
    # Bernoulli, i.e. video-dependent counts occur
    model.train()
    all_counts = set()
    for _ in range(10):
        d = model.preprocess_batch(train_samples[:4])
        all_counts.update(int(x) for x in d[SEL_NUM_SELECTED].tolist())
    check(len(all_counts) >= 2,
          f"variable counts observed across draws/videos: {sorted(all_counts)}")
    check(not (len(counts) == 1 and next(iter(counts)) == 8),
          "selection is NOT fixed-8 (it is a sequential Pick/Drop)")
    print(f"    counts across eval-mode videos: {sorted(counts)}, "
          f"across stochastic draws: {sorted(all_counts)}")


def test3_forward(cfg, train_samples):
    print("=== Test 3: XE forward through selector meta arch ===")
    model = build_model(cfg).cuda().eval()
    data = model.preprocess_batch(train_samples[:4])
    with torch.no_grad():
        out = model(data)
    logits = out["G_LOGITS"]
    check(logits.shape == (4, cfg.MODEL.MAX_SEQ_LEN, cfg.MODEL.VOCAB_SIZE),
          f"G_LOGITS shape {tuple(logits.shape)} == (4, 26, 1891)")
    check(not torch.isnan(logits).any(), "G_LOGITS has no NaN")


def test4_backward(cfg, train_samples):
    print("=== Test 4: backward (XE + budget), selector gradients flow ===")
    model = build_model(cfg).cuda().train()
    losses = build_losses(cfg)

    data = model.preprocess_batch(train_samples[:4])
    out = model(data)
    loss_dict = {}
    for loss_fn in losses:
        loss_dict.update(loss_fn(out))
    xe = sum(loss_dict.values())
    probs = data[SEL_PROBS]
    num_cand = data[SEL_NUM_CANDIDATES].float().clamp(min=1)
    ratio = probs.sum(1) / num_cand
    budget = cfg.SELECTOR.LAMBDA_BUDGET * ((ratio - cfg.SELECTOR.TARGET_RATIO) ** 2).mean()
    total = xe + budget
    check(torch.isfinite(total), f"total loss finite ({total.item():.4f})")
    total.backward()

    g = model.selector.gru.weight_ih.grad
    check(g is not None and torch.isfinite(g).all(), "selector GRU gradients finite")
    check(g.abs().sum().item() > 0, "selector receives nonzero gradient")
    check(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),
          "all model gradients finite")
    opt = build_optimizer(cfg, model)
    before = model.selector.gru.weight_ih.clone()
    opt.step()
    check(not torch.equal(before, model.selector.gru.weight_ih),
          "optimizer.step() updated selector parameters")


def test5_overfit(cfg, mapper_train, train_list):
    print("=== Test 5: 4-video overfit (200 steps) ===")
    unique_ids, picked = [], []
    for entry in train_list:
        vid = entry["video_id"]
        if vid not in unique_ids:
            unique_ids.append(vid)
        if len(unique_ids) <= 4 and vid in unique_ids[:4]:
            picked.append(entry)
    check(len(picked) == 20, f"picked 4 unique videos -> {len(picked)} entries")

    cfg = build_cfg(base_lr=1e-3)
    model = build_model(cfg).cuda().train()
    losses = build_losses(cfg)
    opt = build_optimizer(cfg, model)

    history = []
    for step in range(200):
        idxs = np.random.RandomState(step).choice(len(picked), size=10, replace=False)
        samples = [mapper_train(picked[i]) for i in idxs]
        data = model.preprocess_batch(samples)
        out = model(data)
        xe = sum(l for loss_fn in losses for l in loss_fn(out).values())
        probs = data[SEL_PROBS]
        num_cand = data[SEL_NUM_CANDIDATES].float().clamp(min=1)
        ratio = probs.sum(1) / num_cand
        budget = cfg.SELECTOR.LAMBDA_BUDGET * ((ratio - cfg.SELECTOR.TARGET_RATIO) ** 2).mean()
        total = xe + budget
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_CLIP)
        opt.step()
        history.append(total.item())

    first10 = np.mean(history[:10])
    last10 = np.mean(history[-10:])
    print(f"    loss first10={first10:.4f} last10={last10:.4f} "
          f"(avg selected last step: {data[SEL_NUM_SELECTED].float().mean().item():.1f})")
    check(last10 < 0.7 * first10, f"loss decreased clearly ({first10:.3f} -> {last10:.3f})")


def test6_inference(cfg, test_samples):
    print("=== Test 6: inference + beam search ===")
    model = build_model(cfg).cuda().eval()
    data_gen = model.preprocess_batch(test_samples[:2])
    with torch.no_grad():
        b = model(data_gen, use_beam_search=True, output_sents=True)
    check(len(b["OUTPUT"]) == 2 and all(isinstance(x, str) for x in b["OUTPUT"]),
          "beam(5) decodes 2 caption strings")
    print(f"    beam captions: {b['OUTPUT']}")
    print(f"    num_selected: {[int(x) for x in data_gen[SEL_NUM_SELECTED].tolist()]}")


def test7_coco_eval(cfg):
    print("=== Test 7: COCO evaluation chain on full val (74 videos) ===")
    model = build_model(cfg).cuda().eval()
    loader = build_xmodaler_valtest_loader(cfg, stage="val")
    results = []
    traces = []
    with torch.no_grad():
        for data in tqdm.tqdm(loader):
            data = model.preprocess_batch(data)
            ids = data[kfg.IDS]
            res = model(data, use_beam_search=True, output_sents=True)
            for id, output in zip(ids, res[kfg.OUTPUT]):
                if not isinstance(output, str) or output.strip() == "":
                    output = "."
                results.append({cfg.INFERENCE.ID_KEY: int(id), cfg.INFERENCE.VALUE: output})
            for i, (id, idxs) in enumerate(zip(ids, data[SEL_INDICES])):
                traces.append({"int_id": int(id), "selected_indices": list(idxs),
                               "num_selected": len(idxs)})

    check(len(results) == 74 and len(set(r["image_id"] for r in results)) == 74,
          "74 unique predictions for val split")
    evaluator = build_evaluation(cfg, cfg.INFERENCE.VAL_ANNFILE, None)
    metrics = evaluator.eval(results, -1)
    check("Bleu_4" in metrics and "CIDEr" in metrics,
          f"evaluator returned caption metrics: Bleu_4={metrics.get('Bleu_4'):.3f}, "
          f"CIDEr={metrics.get('CIDEr'):.3f}")
    ns = [t["num_selected"] for t in traces]
    print(f"    selection summary: avg={np.mean(ns):.2f} min={min(ns)} max={max(ns)} "
          f"median={np.median(ns):.1f} ratio={np.mean(ns) / 10:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all", choices=["1", "2", "3", "4", "5", "6", "7", "all"])
    args = parser.parse_args()

    torch.cuda.empty_cache()
    cfg = build_cfg()
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    if args.test in ("1", "all"):
        model, train_samples, test_samples = test1_candidates_and_selector(cfg)
    else:
        train_samples, test_samples, mapper_train, train_list = get_samples(cfg)

    if args.test in ("2", "all"):
        test2_selection_traces(cfg, train_samples)
    if args.test in ("3", "all"):
        test3_forward(cfg, train_samples)
    if args.test in ("4", "all"):
        test4_backward(cfg, train_samples)
    if args.test in ("5", "all"):
        mapper_train = build_dataset_mapper(cfg, name=cfg.DATASETS.TRAIN, stage="train")
        train_list = mapper_train.load_data(cfg)
        test5_overfit(cfg, mapper_train, train_list)
    if args.test in ("6", "all"):
        test6_inference(cfg, test_samples)
    if args.test in ("7", "all"):
        test7_coco_eval(cfg)

    print("\nALL PICKNET-STYLE SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
