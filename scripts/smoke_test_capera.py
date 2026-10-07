"""CapERA XE baseline smoke tests (Phase 1 gate).

Test 1: dataset loading (2-4 videos)      -> ATT_FEATS/ATT_MASKS/token protocol
Test 2: full forward + greedy + beam gen  -> G_LOGITS shape, no NaN, captions decode
Test 3: backward                          -> finite loss/grads, optimizer.step works
Test 4: 4-video overfit (200 steps)       -> training loss decreases clearly

Usage: python scripts/smoke_test_capera.py [--test 1|2|3|4|all] [--skip-beam]
All tests are self-contained; they never touch experiments/ checkpoints or
official annotations.
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XMODALER_DIR = os.path.join(PROJECT_ROOT, "third_party", "xmodaler")
sys.path.insert(0, XMODALER_DIR)

import xmodaler.datasets.videos.capera  # noqa: F401  (register CapERADataset)

import numpy as np
import torch

from xmodaler.config import get_cfg
from xmodaler.datasets import build_dataset_mapper
from xmodaler.losses import build_losses
from xmodaler.modeling import add_config, build_model
from xmodaler.optim import build_optimizer
from xmodaler.utils.env import seed_all_rng

CONFIG = os.path.join(PROJECT_ROOT, "configs", "capera", "xe_baseline.yaml")


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


def test1_dataset(cfg):
    print("=== Test 1: dataset loading (train x2, test x2) ===")
    mapper_train = build_dataset_mapper(cfg, name=cfg.DATASETS.TRAIN, stage="train")
    train_list = mapper_train.load_data(cfg)
    samples = [mapper_train(train_list[i]) for i in range(2)]

    for s in samples:
        feats = s["ATT_FEATS"]
        check(feats.shape == (cfg.DATALOADER.MAX_FEAT_NUM, 768),
              f"ATT_FEATS shape {tuple(feats.shape)} == (8, 768)")
        check(torch.is_tensor(feats) and feats.dtype == torch.float32, "ATT_FEATS float32 tensor")
        t = s["G_TOKENS_IDS"][0]
        check(t.ndim == 1 and t[0].item() == 0, "G_TOKENS_IDS starts with BOS id 0")
        tg = s["G_TARGET_IDS"][0]
        check(0 in tg and -1 in tg, "G_TARGET_IDS contains EOS(0) and -1 padding")
        check(0 <= s["IDS"] < 1473, f"train IDS {s['IDS']} in [0,1473)")

    mapper_test = build_dataset_mapper(cfg, name=cfg.DATASETS.TEST, stage="test")
    test_list = mapper_test.load_data(cfg)
    t_samples = [mapper_test(test_list[i]) for i in range(2)]
    for s in t_samples:
        check(s["ATT_FEATS"].shape == (cfg.DATALOADER.MAX_FEAT_NUM, 768),
              f"test ATT_FEATS shape {tuple(s['ATT_FEATS'].shape)} == (8, 768)")
        check(s["IDS"] >= 1473, f"test IDS {s['IDS']} >= 1473 (split identity)")
        check("G_TOKENS_IDS" not in s, "test mode has no tokens (generation mode)")

    return samples, t_samples, mapper_train, train_list


def test2_forward(cfg, samples, t_samples, skip_beam):
    print("=== Test 2: forward + generation ===")
    model = build_model(cfg).cuda().eval()

    data = model.preprocess_batch(samples)
    with torch.no_grad():
        out = model(data)
    logits = out["G_LOGITS"]
    check(logits.shape == (2, cfg.MODEL.MAX_SEQ_LEN, cfg.MODEL.VOCAB_SIZE),
          f"G_LOGITS shape {tuple(logits.shape)} == (2, 26, 1891)")
    check(not torch.isnan(logits).any(), "G_LOGITS has no NaN")

    data_gen = model.preprocess_batch(t_samples)
    with torch.no_grad():
        g = model(data_gen, use_beam_search=False, output_sents=True)
    check(g["G_SENTS_IDS"].shape == (2, cfg.MODEL.MAX_SEQ_LEN), "greedy G_SENTS_IDS (2, 26)")
    check(len(g["OUTPUT"]) == 2 and all(isinstance(x, str) for x in g["OUTPUT"]),
          "greedy decodes 2 caption strings")
    print(f"    greedy captions: {g['OUTPUT']}")

    if not skip_beam:
        # NOTE: greedy decode mutates the input dict in place (upstream xmodaler
        # behavior: inputs.update(ve_out) replaces ATT_FEATS), so rebuild the
        # batch before beam search.
        with torch.no_grad():
            b = model(model.preprocess_batch(t_samples), use_beam_search=True, output_sents=True)
        check(len(b["OUTPUT"]) == 2 and all(isinstance(x, str) for x in b["OUTPUT"]),
              "beam(5) decodes 2 caption strings")
        print(f"    beam captions: {b['OUTPUT']}")
    return model


def test3_backward(cfg, samples, model):
    print("=== Test 3: backward ===")
    losses = build_losses(cfg)
    model = model.train()

    data = model.preprocess_batch(samples)
    out = model(data)
    loss_dict = {}
    for loss_fn in losses:
        loss_dict.update(loss_fn(out))
    total = sum(loss_dict.values())
    check(torch.isfinite(total), f"loss finite ({total.item():.4f})")
    print(f"    losses: { {k: round(v.item(), 4) for k, v in loss_dict.items()} }")

    total.backward()
    grads_finite = all(torch.isfinite(p.grad).all() for p in model.parameters()
                       if p.grad is not None)
    check(grads_finite, "all gradients finite")

    opt = build_optimizer(cfg, model)
    before = {n: p.clone() for n, p in model.named_parameters()}
    opt.step()
    changed = any(not torch.equal(before[n], p) for n, p in model.named_parameters())
    check(changed, "optimizer.step() updated parameters")


def test4_overfit(cfg, mapper_train, train_list):
    print("=== Test 4: 4-video overfit (200 steps) ===")
    # 4 unique videos x 5 captions = 20 entries
    unique_ids, picked = [], []
    for entry in train_list:
        vid = entry["video_id"]
        if vid not in unique_ids:
            unique_ids.append(vid)
        if len(unique_ids) <= 4 and vid in unique_ids[:4]:
            picked.append(entry)
    check(len(unique_ids) >= 4 and len(picked) == 20,
          f"picked 4 unique videos -> {len(picked)} caption entries")

    cfg = build_cfg(base_lr=1e-3)  # fresh cfg + higher lr for fast overfit
    model = build_model(cfg).cuda().train()
    losses = build_losses(cfg)
    opt = build_optimizer(cfg, model)

    batch_size = 10
    history = []
    for step in range(200):
        idxs = np.random.RandomState(step).choice(len(picked), size=batch_size, replace=False)
        samples = [mapper_train(picked[i]) for i in idxs]
        data = model.preprocess_batch(samples)
        out = model(data)
        total = sum(l for loss_fn in losses for l in loss_fn(out).values())
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_CLIP)
        opt.step()
        history.append(total.item())

    first10 = np.mean(history[:10])
    last10 = np.mean(history[-10:])
    print(f"    loss first10={first10:.4f} last10={last10:.4f}")
    check(last10 < 0.7 * first10,
          f"training loss decreased clearly ({first10:.3f} -> {last10:.3f})")
    check(last10 < 2.0, f"overfit loss low enough ({last10:.3f} < 2.0)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all", choices=["1", "2", "3", "4", "all"])
    parser.add_argument("--skip-beam", action="store_true", help="skip beam-search generation check")
    args = parser.parse_args()

    torch.cuda.empty_cache()
    cfg = build_cfg()
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    samples, t_samples, mapper_train, train_list = None, None, None, None
    model = None

    if args.test in ("1", "all"):
        samples, t_samples, mapper_train, train_list = test1_dataset(cfg)
    else:
        mapper_train = build_dataset_mapper(cfg, name=cfg.DATASETS.TRAIN, stage="train")
        train_list = mapper_train.load_data(cfg)

    if args.test in ("2", "all"):
        if samples is None:
            samples, t_samples, mapper_train, train_list = test1_dataset(cfg)
        model = test2_forward(cfg, samples, t_samples, args.skip_beam)

    if args.test in ("3", "all"):
        if model is None:
            if samples is None:
                samples, t_samples, mapper_train, train_list = test1_dataset(cfg)
            model = build_model(cfg).cuda().eval()
        test3_backward(cfg, samples, model)

    if args.test in ("4", "all"):
        test4_overfit(cfg, mapper_train, train_list)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
