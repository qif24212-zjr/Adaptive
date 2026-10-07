# -*- coding: utf-8 -*-
"""
Train the Adaptive Frame Pruning pipeline (REINFORCE + utility critic).

    candidate frames -> lightweight encoder -> AdaptiveSelector (SELECT/STOP)
        -> heavy CLIP on SELECTED frames only -> frozen captioner
        -> CIDEr-based reward -> policy/critic update

Usage:
    python scripts/train_adaptive_frame_pruning.py \
        --config configs/adaptive_frame_pruning.yaml \
        [--epochs N] [--batch-size N] [--max-batches N] [--limit N]

NOTE: the CoCap caption head currently has NO trained checkpoint (it is only
CLIP-initialized), so CIDEr rewards are NOT meaningful yet — this script
verifies TRAINING CORRECTNESS (finite losses/gradients, checkpoints) and is
ready for a real captioner checkpoint via `captioner.checkpoint` in config.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "xmodaler"))

import torch
import yaml

from models.frame_cocap import build_frame_captioner, read_video_frames_cv2
from models.selectors.adaptive_frame_pruning import AdaptiveFramePruning, CountingEncoder
from models.selectors.adaptive_selector import AdaptiveSelector
from models.selectors.caption_reward import CaptionReward
from models.selectors.lightweight_frame_encoder import LightweightFrameEncoder
from models.selectors.policy_trainer import SelectorPolicyTrainer, load_captioner_checkpoint
from models.selectors.utility_critic import UtilityCritic

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_all(cfg):
    torch.manual_seed(cfg["training"]["seed"])
    torch.cuda.manual_seed(cfg["training"]["seed"])
    random.seed(cfg["training"]["seed"])

    captioner = build_frame_captioner(
        frame_encoder=cfg["frame_encoder"], clip_path=cfg["clip_path"],
        max_t_len=cfg["captioner"].get("max_t_len", 77),
    ).to(DEVICE).eval()
    for p in captioner.parameters():
        p.requires_grad_(False)
    if cfg["captioner"].get("checkpoint"):
        print(f"[captioner] loading checkpoint {cfg['captioner']['checkpoint']}")
        load_captioner_checkpoint(captioner, cfg["captioner"]["checkpoint"], DEVICE)

    selector = AdaptiveSelector(
        feature_dim=cfg["selector"]["feature_dim"],
        hidden_size=cfg["selector"]["hidden_size"],
        min_selected_frames=cfg["min_selected_frames"],
        max_selected_frames=cfg["max_selected_frames"],
        max_candidates=cfg["selector"]["max_candidates"],
        sample=cfg["selector"]["sample"],
    ).to(DEVICE)

    lightweight = LightweightFrameEncoder(
        feature_dim=cfg["lightweight_encoder"]["feature_dim"],
        width=cfg["lightweight_encoder"]["width"],
    ).to(DEVICE)

    pruning = AdaptiveFramePruning(
        lightweight_encoder=lightweight,
        selector=selector,
        heavy_encoder=CountingEncoder(captioner.frame_encoder),
        captioner=captioner,
    )

    critic = UtilityCritic(input_dim=cfg["selector"]["hidden_size"],
                           hidden_dim=cfg["critic"]["hidden_dim"]).to(DEVICE)

    # corpus-level DF for meaningful per-sample CIDEr (SCST-style);
    # without it, tiny-batch CIDEr is structurally ~0 (idf=log(1)=0)
    corpus_refs = {}
    for split in ("train", "val", "test"):
        a = json.load(open(f"data/CapERA/captions_{split}.json"))
        for x in a["annotations"]:
            corpus_refs.setdefault(f"{split}_{x['image_id']}", []).append(x["caption"])

    reward = CaptionReward(
        mode=cfg["reward"]["mode"], metric=cfg["reward"]["metric"],
        lambda_cost=cfg["reward"]["lambda_cost"], gamma=cfg["reward"]["gamma"],
        max_selected_frames=cfg["max_selected_frames"],
        corpus_refs=corpus_refs,
    )

    trainer = SelectorPolicyTrainer(
        pruning=pruning, captioner=captioner, critic=critic, reward=reward,
        lr_selector=cfg["training"]["lr_selector"],
        lr_critic=cfg["training"]["lr_critic"],
        lambda_utility=cfg["critic"]["lambda_utility"],
        gamma=cfg["reward"]["gamma"],
        lr_decay_gamma=cfg["training"].get("lr_decay_gamma", 0.95),
        device=torch.device(DEVICE),
    )
    return trainer, captioner, pruning, critic


def load_video_batch(cfg, images, refs, video_index, cache, start, batch_size):
    """Frames come from a disk cache when available (predecoded uint8),
    else are decoded once and saved there; hot in RAM afterwards."""
    frames_list, refs_list = [], []
    cache_dir = Path(cfg["data"].get("frames_cache_dir", "")) \
        if cfg["data"].get("frames_cache_dir") else None
    for img in images[start:start + batch_size]:
        vid = img["id"]
        if vid not in cache:
            disk_path = cache_dir / f"{vid}.pt" if cache_dir else None
            if disk_path is not None and disk_path.exists():
                f = torch.load(disk_path, map_location="cpu")
            else:
                path = video_index[img["file_name"].replace(" ", "")]
                f = read_video_frames_cv2(str(path), n_frames=cfg["candidate_frames"],
                                          sample="uniform")
                assert f is not None and f.shape[0] == cfg["candidate_frames"], img["file_name"]
                if disk_path is not None:
                    # store the EXACT normalized float frames the model sees
                    disk_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(f, disk_path)
            cache[vid] = f  # unbounded RAM cache (host has ~1TB RAM)
        frames_list.append(cache[vid])
        refs_list.append(refs[vid])
    return torch.stack(frames_list).to(DEVICE), refs_list


def run_content_probe(pruning, probe_frames, M):
    """Fixed 20-video content-dependence probe (argmax, no_grad — no weight
    change): per-step top-1 consistency, probability cross-video std, STOP
    probability cross-video std, plus the argmax pattern distribution.
    Reuses the instrumented replica already verified against the real
    selector (conditioning_fix_check.py)."""
    sys.path.insert(0, str(Path("experiments/adaptive_main_epoch20").resolve()))
    from conditioning_fix_check import instrumented_episode
    by_step, stop_steps, patterns = {}, Counter(), Counter()
    with torch.no_grad():
        for vid, f in probe_frames.items():
            lw = pruning.lightweight_encoder(
                f.to(DEVICE).reshape(-1, 3, 224, 224)).reshape(M, -1)
            m = instrumented_episode(pruning.selector, lw, mode="argmax")
            for r in m["records"]:
                by_step.setdefault(r["step"], []).append(r)
            stop_steps[m["stop_step"]] += 1
            patterns[tuple(m["indices"])] += 1
    out = {}
    for t, rows in sorted(by_step.items()):
        n = len(rows)
        probs = torch.tensor([r["probs"] for r in rows], dtype=torch.float32)
        tops = [r["argmax_action"] for r in rows]
        top1 = max(set(tops), key=tops.count)
        stops = torch.tensor([r["stop_prob"] for r in rows], dtype=torch.float32)
        out[str(t)] = {
            "n_videos": n,
            "top1_consistency": tops.count(top1) / n,
            "top1_action": top1,
            "prob_cross_video_std_mean":
                round(float(probs.std(dim=0).mean()), 6) if n >= 2 else None,
            "stop_prob_mean": round(float(stops.mean()), 6),
            "stop_prob_cross_video_std":
                round(float(stops.std()), 6) if n >= 2 else None,
        }
    return {"per_step": out,
            "stop_step_distribution": {str(k): v for k, v in sorted(stop_steps.items())},
            "n_distinct_patterns": len(patterns),
            "top_patterns": {",".join(map(str, k)): v for k, v in patterns.most_common(5)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adaptive_frame_pruning.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap dataset videos (dry-run)")
    parser.add_argument("--resume", default=None, help="resume from a trainer checkpoint (last.pt)")
    parser.add_argument("--captioner-checkpoint", default=None,
                        help="override captioner.checkpoint")
    parser.add_argument("--output-dir", default=None, help="override output_dir")
    parser.add_argument("--limit-seed", type=int, default=42,
                        help="seed for deterministic --limit subset selection")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.epochs: cfg["training"]["epochs"] = args.epochs
    if args.batch_size: cfg["training"]["batch_size"] = args.batch_size
    if args.max_batches is not None: cfg["training"]["max_batches_per_epoch"] = args.max_batches
    if args.captioner_checkpoint: cfg["captioner"]["checkpoint"] = args.captioner_checkpoint
    if args.output_dir: cfg["output_dir"] = args.output_dir
    cfg = OrderedDict(cfg)

    out_dir = Path(cfg["output_dir"])
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(args.config, out_dir / "config_snapshot.yaml")

    print("=" * 60)
    print("Adaptive Frame Pruning Training (FORMAL RUN)")
    print(f"  config: {args.config} -> snapshot at {out_dir / 'config_snapshot.yaml'}")
    print(f"  output_dir: {out_dir}")
    print(f"  M={cfg['candidate_frames']} min={cfg['min_selected_frames']} "
          f"max={cfg['max_selected_frames']} sample={cfg['selector']['sample']}")
    print(f"  reward: mode={cfg['reward']['mode']} lambda_cost={cfg['reward']['lambda_cost']} "
          f"gamma={cfg['reward']['gamma']}")
    print(f"  captioner: frozen={cfg['captioner']['freeze']} "
          f"checkpoint={cfg['captioner'].get('checkpoint')}")
    print(f"  training: epochs={cfg['training']['epochs']} batch={cfg['training']['batch_size']} "
          f"lr_selector={cfg['training']['lr_selector']} lr_critic={cfg['training']['lr_critic']}")
    print("=" * 60)

    trainer, captioner, pruning, critic = build_all(cfg)

    # ---- preflight asserts (formal-run invariants) ----
    assert all(not p.requires_grad for p in captioner.parameters()), \
        "captioner has trainable parameters!"
    opt_param_ids = set()
    for group in trainer.opt_selector.param_groups + trainer.opt_critic.param_groups:
        opt_param_ids.update(id(p) for p in group["params"])
    cap_param_ids = {id(p) for p in captioner.parameters()}
    assert opt_param_ids.isdisjoint(cap_param_ids), "captioner params leaked into optimizers!"
    for module, label in ((pruning.selector, "selector"),
                          (pruning.lightweight_encoder, "lightweight_encoder"),
                          (critic, "critic")):
        ids = {id(p) for p in module.parameters()}
        assert ids <= opt_param_ids, f"{label} params missing from optimizers!"
    assert trainer.pruning.selector.sample, "training must use stochastic sampling"
    assert trainer.reward._corpus is not None, "reward must use CorpusCider (corpus DF)"
    assert trainer.step_wise_reward, "step-wise dCIDEr reward must be enabled"
    assert trainer.advantage_normalization, "advantage normalization must be enabled"
    assert cfg["candidate_frames"] == 32 and cfg["min_selected_frames"] == 2 \
        and cfg["max_selected_frames"] == 16, "M/min/max mismatch!"
    assert cfg["reward"]["lambda_cost"] == 0.01, "lambda_c mismatch!"
    print("[preflight] captioner frozen: OK | optimizers: captioner disjoint + "
          "selector/lightweight/critic included: OK | stochastic sampling: OK | "
          "CorpusCider: OK | step-wise reward: OK | advantage norm: OK | "
          "M=32 min=2 max=16 lambda=0.01: OK")

    start_epoch = 1
    if args.resume:
        state = trainer.load_checkpoint(args.resume)
        start_epoch = int(state["epoch"]) + 1
        print(f"[resume] {args.resume} -> epoch {start_epoch}")

    # ---- data ----
    ann = json.load(open(cfg["data"]["annotation"]))
    refs = {}
    for x in ann["annotations"]:
        refs.setdefault(x["image_id"], []).append(x["caption"])
    images = ann["images"]
    if args.limit:
        # deterministic subset (same seed-42 protocol as the diagnostics)
        r = random.Random(args.limit_seed)
        r.shuffle(images)
        images = images[:args.limit]
    else:
        random.shuffle(images)
    video_index = {}
    for p in Path(cfg["data"]["video_root"]).rglob("*.mp4"):
        video_index[p.name.replace(" ", "")] = p
    # a few CapERA videos have <32 frames (no cache entry); they cannot
    # provide the candidate pool and are excluded from formal training
    cache_dir = Path(cfg["data"].get("frames_cache_dir", "")) \
        if cfg["data"].get("frames_cache_dir") else None
    if cache_dir:
        before = len(images)
        images = [img for img in images if (cache_dir / f"{img['id']}.pt").exists()]
        if before != len(images):
            print(f"[data] excluded {before - len(images)} short videos without cached frames")
    print(f"[data] {len(images)} videos, {len(refs)} with refs")

    # ---- fixed 20-video content probe (same videos as the conditioning
    #      diagnostics; runs after every epoch, never changes weights) ----
    probe_ann = json.load(open("data/CapERA/captions_test.json"))
    probe_img = {str(x["id"]): x for x in probe_ann["images"]}
    probe_vi = {p.name.replace(" ", ""): p
                for p in Path("datasets/CapERA/videos/Videos/Test").rglob("*.mp4")}
    probe_vids = json.load(open(
        "experiments/adaptive_main_epoch20/selector_content_dependency_diagnostic.json")
    )["setup"]["video_ids"]
    probe_frames = {}
    for vid in probe_vids:
        p = probe_vi[probe_img[vid]["file_name"].replace(" ", "")]
        f = read_video_frames_cv2(str(p), n_frames=cfg["candidate_frames"], sample="uniform")
        assert f is not None and f.shape[0] == cfg["candidate_frames"], vid
        probe_frames[vid] = f
    print(f"[probe] {len(probe_frames)} fixed test videos ready")

    B = cfg["training"]["batch_size"]
    batches_per_epoch = min(len(images) // B, cfg["training"]["max_batches_per_epoch"]
                            or (len(images) // B))
    step_log_file = out_dir / "training_log.json"
    best_key = None
    cache = {}
    global_step = 0

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        random.shuffle(images)
        agg = {k: 0.0 for k in ("policy_loss", "critic_loss", "total_loss", "mean_reward",
                                "mean_return", "mean_advantage_raw",
                                "mean_advantage_normalized", "advantage_std",
                                "positive_reward_ratio",
                                "mean_cider", "mean_selected_frames", "selection_ratio",
                                "mean_stop_step", "grad_norm")}
        min_sel, max_sel = 10 ** 9, -1
        hist, patterns = Counter(), Counter()
        all2 = all16 = 0.0
        t0 = time.time()
        for bi in range(batches_per_epoch):
            frames, refs_b = load_video_batch(cfg, images, refs, video_index, cache,
                                              bi * B, B)
            m = trainer.train_step(frames, refs_b)
            global_step += 1
            for k in agg: agg[k] += m[k]
            min_sel = min(min_sel, m["min_selected"])
            max_sel = max(max_sel, m["max_selected"])
            for c, cnt in m["selected_count_distribution"].items():
                hist[c] += cnt
            all2 += m["selected_count_distribution"].get("2", 0)
            all16 += m["selected_count_distribution"].get("16", 0)
            for p in m["patterns"]:
                patterns[p] += 1
            with open(step_log_file, "a") as f:
                f.write(json.dumps({"epoch": epoch, "step": global_step, **m}) + "\n")
            if global_step % 500 == 0:
                trainer.save_checkpoint(str(out_dir / "checkpoints" / "last.pt"),
                                        epoch, dict(cfg))
        for k in agg: agg[k] /= max(batches_per_epoch, 1)
        trainer.scheduler_step()

        # ---- per-epoch health checks ----
        import math as _math
        assert all(_math.isfinite(v) for v in agg.values()), f"non-finite metric: {agg}"
        if agg["mean_selected_frames"] < 2.05:
            print("!!! HEALTH WARNING: selected count collapsed to all-2 "
                  "(no auto changes; stopping criterion for the operator)", flush=True)
        if agg["mean_selected_frames"] > 15.95:
            print("!!! HEALTH WARNING: selected count pinned at all-16", flush=True)
        if abs(agg["mean_reward"]) < 1e-9:
            print("!!! HEALTH WARNING: reward looks constant", flush=True)

        # epoch stats per the spec
        record = {
            "epoch": epoch,
            **agg,
            "min_selected_count": min_sel,
            "max_selected_count": max_sel,
            "mean_pruning_ratio": 1.0 - agg["selection_ratio"],
            "epoch_time_s": round(time.time() - t0, 1),
            "selected_count_histogram": {k: hist[k] for k in sorted(hist, key=int)},
            "all2_fraction": round(all2 / max(batches_per_epoch * B, 1), 4),
            "all16_fraction": round(all16 / max(batches_per_epoch * B, 1), 4),
            "n_distinct_patterns": len(patterns),
            "top_patterns": {p: n for p, n in patterns.most_common(5)},
        }
        with open(out_dir / "logs" / "epoch_log.json", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[epoch {epoch}] policy_loss={agg['policy_loss']:.4f} "
              f"critic_loss={agg['critic_loss']:.4f} total={agg['total_loss']:.4f} | "
              f"reward={agg['mean_reward']:.4f} return={agg['mean_return']:.4f} "
              f"adv_raw={agg['mean_advantage_raw']:.3f} "
              f"adv_norm={agg['mean_advantage_normalized']:.3f} "
              f"adv_std={agg['advantage_std']:.3f} pos_r={agg['positive_reward_ratio']:.2f} | "
              f"cider={agg['mean_cider']:.3f} | "
              f"selected: mean={agg['mean_selected_frames']:.2f} "
              f"min={min_sel} max={max_sel} pruning={record['mean_pruning_ratio']:.3f} | "
              f"grad_norm={agg['grad_norm']:.2f} lr={m['lr_selector']:.2e}",
              flush=True)

        # checkpoints
        trainer.save_checkpoint(str(out_dir / "checkpoints" / "last.pt"), epoch, dict(cfg))
        key = (agg["mean_cider"], -agg["mean_selected_frames"])
        if best_key is None or key > best_key:
            best_key = key
            trainer.save_checkpoint(str(out_dir / "checkpoints" / "best.pt"), epoch, dict(cfg),
                                    extra={"mean_cider": agg["mean_cider"],
                                           "mean_selected_frames": agg["mean_selected_frames"]})
            print(f"    -> new best (mean_cider={agg['mean_cider']:.4f}, "
                  f"mean_selected={agg['mean_selected_frames']:.2f})", flush=True)

        # ---- fixed-video content probe (re-collapse monitoring) ----
        probe = run_content_probe(pruning, probe_frames, cfg["candidate_frames"])
        with open(out_dir / "logs" / "content_probe.json", "a") as f:
            f.write(json.dumps({"epoch": epoch, **probe}) + "\n")
        print(f"[probe e{epoch}] patterns={probe['n_distinct_patterns']} "
              f"stop_steps={probe['stop_step_distribution']} | top1=" +
              ",".join(f"{t}:{v['top1_consistency']:.2f}"
                       for t, v in probe["per_step"].items()) + " | prob_std=" +
              ",".join(f"{t}:{v['prob_cross_video_std_mean']}"
                       for t, v in probe["per_step"].items()), flush=True)

    # final summary.json (last-step metrics + aggregate stats)
    import statistics as _st
    lines = [json.loads(l) for l in open(step_log_file)]
    summary = {
        "total_steps": len(lines),
        "final": lines[-1] if lines else {},
        "mean_selected_frames_over_last_10_steps": round(
            _st.mean([e["mean_selected_frames"] for e in lines[-10:]]), 3),
        "mean_end_at_2_over_last_10_steps": round(
            _st.mean([e["episode_length_fractions"]["end_at_2"] for e in lines[-10:]]), 3),
        "mean_reward_last_10": round(
            _st.mean([e["mean_reward"] for e in lines[-10:]]), 3),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] checkpoints in {out_dir / 'checkpoints'}, step log in {step_log_file}, "
          f"summary in {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
