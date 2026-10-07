# -*- coding: utf-8 -*-
"""
Train the Frame-CoCap captioner on CapERA (supervised, teacher forcing).

    CapERA train video -> uniform 32 frames -> frozen CLIP ViT-B/16
        -> [B, 32, 512] -> FrameCaptionHead (trainable) -> CE loss

Modes:
    --overfit N          tiny overfit on N train videos (first sanity gate)
    (default)            1-epoch+ training on cached CLIP features

Checkpoints: experiments/frame_cocap/checkpoints/{best,last}.pt in the
format the Adaptive pipeline's captioner hook loads.

Usage:
    python scripts/train_frame_cocap.py --config configs/frame_captioning_training.yaml
    python scripts/train_frame_cocap.py --overfit 16
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "xmodaler"))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "CoCap"))

import torch
import yaml

from models.frame_cocap import build_frame_captioner, generate_caption, read_video_frames_cv2
from models.frame_cocap.training import CaptionerTrainer
from cocap.modeling.eval_captioning import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/frame_captioning_training.yaml")
    parser.add_argument("--overfit", type=int, default=None, help="tiny overfit on N train videos")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--use-precomputed", action="store_true",
                        help="load features from config data.features_dir")
    parser.add_argument("--resume", default=None,
                        help="resume from a saved trainer checkpoint (e.g. best.pt)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.epochs: cfg["training"]["epochs"] = args.epochs
    if args.batch_size: cfg["training"]["batch_size"] = args.batch_size
    if args.log_every: cfg["training"]["log_every_steps"] = args.log_every
    if args.lr: cfg["training"]["lr"] = args.lr

    torch.manual_seed(cfg["training"]["seed"])
    random.seed(cfg["training"]["seed"])
    out_dir = Path(cfg["output_dir"])
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Frame-CoCap captioner training (supervised)")
    print(f"  encoder: {cfg['frame_encoder']} (frozen={cfg['freeze_frame_encoder']})")
    print(f"  num_frames: {cfg['num_frames']} | batch: {cfg['training']['batch_size']}")
    print("=" * 60)

    captioner = build_frame_captioner(frame_encoder=cfg["frame_encoder"],
                                      clip_path=cfg["clip_path"]).to(DEVICE)
    trainer = CaptionerTrainer(
        captioner, lr=cfg["training"]["lr"],
        lr_pretrained_embeddings=cfg["training"]["lr_pretrained_embeddings"],
        label_smoothing=cfg["training"]["label_smoothing"],
        freeze_frame_encoder=cfg["freeze_frame_encoder"],
    )
    start_epoch = 1
    if args.resume:
        state = trainer.load(args.resume, device=DEVICE)
        start_epoch = int(state["epoch"]) + 1
        print(f"[resume] {args.resume} -> starting epoch {start_epoch}")

    # ---- data ----
    train_ann = json.load(open(cfg["data"]["train_annotation"]))
    val_ann = json.load(open(cfg["data"]["val_annotation"]))
    refs = {}
    for x in train_ann["annotations"]:
        refs.setdefault(x["image_id"], []).append(x["caption"])
    val_refs = {}
    for x in val_ann["annotations"]:
        val_refs.setdefault(x["image_id"], []).append(x["caption"])
    train_images = train_ann["images"]
    feat_dir = Path(cfg["data"]["features_dir"]) / "train"
    if args.use_precomputed:
        # a few CapERA videos have <32 frames (no precomputed features);
        # they cannot provide the candidate pool and are excluded
        missing = [img for img in train_images if not (feat_dir / f"{img['id']}.pt").exists()]
        if missing:
            print(f"[data] excluding {len(missing)} short videos without precomputed "
                  f"features, e.g. {missing[0]['file_name']}")
        train_images = [img for img in train_images if (feat_dir / f"{img['id']}.pt").exists()]
    if args.overfit:
        random.shuffle(train_images)
        train_images = train_images[:args.overfit]
    random.shuffle(train_images)

    # video index + feature cache
    video_root = Path(cfg["data"]["train_video_root"])
    video_index = {p.name.replace(" ", ""): p for p in video_root.rglob("*.mp4")}
    feat_cache = {}

    def features_for(img):
        vid = img["id"]
        if vid not in feat_cache:
            f_path = feat_dir / f"{vid}.pt"
            if args.use_precomputed and f_path.exists():
                feat_cache[vid] = torch.load(f_path, map_location=DEVICE)
            else:
                path = video_index[img["file_name"].replace(" ", "")]
                frames = read_video_frames_cv2(str(path), n_frames=cfg["num_frames"],
                                               sample="uniform")
                assert frames is not None and frames.shape[0] == cfg["num_frames"], img["file_name"]
                frames = frames.to(DEVICE)
                with torch.no_grad():
                    feats = torch.cat([captioner.frame_encoder(frames[i:i + CHUNK])
                                       for i in range(0, frames.shape[0], CHUNK)], dim=0)
                feat_cache[vid] = feats
        return feat_cache[vid]

    def eval_split(images, refs_dict, tag):
        trainer.eval()
        preds, times = {}, []
        for img in images:
            with torch.no_grad():
                feats = features_for(img).unsqueeze(0)
                mask = torch.ones(1, cfg["num_frames"], dtype=torch.long, device=DEVICE)
                cap = generate_caption(captioner, visual_features=feats, frame_mask=mask)[0]
            preds[str(img["id"])] = [{"sentence": cap}]
        refs_sub = {str(k): v for k, v in refs_dict.items() if str(k) in preds}
        metrics = {k: round(v * 100, 2) for k, v in
                   evaluate(submission={"results": preds}, reference=refs_sub).items()
                   if isinstance(v, float)}
        print(f"    [{tag}] n={len(preds)} CIDEr={metrics.get('CIDEr')} "
              f"Bleu_4={metrics.get('Bleu_4')} METEOR={metrics.get('METEOR')} "
              f"ROUGE_L={metrics.get('ROUGE_L')}")
        return metrics, preds

    B = cfg["training"]["batch_size"]
    best_cider = -1.0
    if args.resume:
        try:
            prog = json.load(open(out_dir / "captioner_progression.json"))
            best_cider = max((e.get("val_cider") or -1.0 for e in prog), default=-1.0)
            print(f"[resume] previous best val CIDEr: {best_cider}")
        except Exception:
            pass
    global_step = 0

    # progression evaluation machinery (only at save_epochs)
    save_epochs = set(cfg["training"].get("save_epochs", []))
    if save_epochs:
        from models.frame_cocap.sensitivity_eval import (
            build_corpus_reward, build_video_index, evaluate_frame_sensitivity,
            load_sensitivity_images)
        sens_images, sens_refs = load_sensitivity_images(20)
        sens_reward = build_corpus_reward()
        sens_index = build_video_index()

        def run_progression_eval(epoch, val_metrics):
            trainer.eval()
            stats = evaluate_frame_sensitivity(trainer.captioner, sens_images, sens_refs,
                                               sens_reward, sens_index, device=DEVICE)
            entry = {
                "epoch": epoch,
                "checkpoint": f"experiments/frame_cocap/checkpoints/epoch_{epoch:02d}.pt",
                "val_cider": val_metrics.get("CIDEr"),
                "val_bleu4": val_metrics.get("Bleu_4"),
                "val_meteor": val_metrics.get("METEOR"),
                "val_rougeL": val_metrics.get("ROUGE_L"),
                **stats,
            }
            prog_file = out_dir / "captioner_progression.json"
            entries = json.load(open(prog_file)) if prog_file.exists() else []
            entries = [e for e in entries if e["epoch"] != epoch]
            entries.append(entry)
            entries.sort(key=lambda e: e["epoch"])
            with open(prog_file, "w") as f:
                json.dump(entries, f, indent=2)
            print(f"[progression] epoch {epoch}: {json.dumps(entry)}", flush=True)
            trainer.train()

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        trainer.train()
        epoch_loss, n_steps = 0.0, 0
        random.shuffle(train_images)
        t0 = time.time()
        for i in range(0, len(train_images), B):
            batch_imgs = train_images[i:i + B]
            feats = torch.stack([features_for(img) for img in batch_imgs])          # (B,32,512)
            mask = torch.ones(feats.shape[0], cfg["num_frames"], dtype=torch.long,
                              device=DEVICE)
            captions = [random.choice(refs[img["id"]]) for img in batch_imgs]
            loss = trainer.train_step(feats, mask, captions)
            loss.backward()
            trainer.optimizer_step()
            epoch_loss += float(loss.detach())
            n_steps += 1
            global_step += 1

            if global_step % cfg["training"]["log_every_steps"] == 0:
                print(f"[epoch {epoch} step {global_step}] loss={loss.item():.4f} "
                      f"(avg {epoch_loss / n_steps:.4f}) elapsed {time.time() - t0:.0f}s",
                      flush=True)
                if args.overfit and global_step % (cfg["training"]["log_every_steps"] * 2) == 0:
                    # overfit gate: training CIDEr + sample captions
                    m, preds = eval_split(train_images[:args.overfit], refs, "overfit-train")
                    for img in train_images[:2]:
                        print(f"      [{img['file_name']}] '{preds[str(img['id'])][0]['sentence']}'")
                        print(f"      gt: '{refs[img['id']][0]}'")
                    trainer.train()  # restore training mode after eval

            if cfg["training"].get("eval_every_steps") and \
                    global_step % cfg["training"]["eval_every_steps"] == 0:
                val_metrics, _ = eval_split(val_ann["images"], val_refs, "val")
                trainer.train()
                if val_metrics.get("CIDEr", -1) > best_cider:
                    best_cider = val_metrics["CIDEr"]
                    trainer.save(str(out_dir / "checkpoints" / "best.pt"), epoch, cfg,
                                 extra={"val_cider": best_cider, "step": global_step})

        trainer.scheduler_step()
        avg = epoch_loss / max(n_steps, 1)
        print(f"[epoch {epoch}] train_loss={avg:.4f} ({time.time() - t0:.0f}s)", flush=True)
        # per-epoch val eval only in normal (non-overfit) mode; overfit uses
        # the step-based overfit-train gate instead (val eval is slow: it
        # decodes 74 videos per call)
        if not args.overfit:
            val_metrics, _ = eval_split(val_ann["images"], val_refs, "val")
            trainer.train()
            if val_metrics.get("CIDEr", -1) > best_cider:
                best_cider = val_metrics["CIDEr"]
                trainer.save(str(out_dir / "checkpoints" / "best.pt"), epoch, cfg,
                             extra={"val_cider": best_cider, "step": global_step})
            trainer.save(str(out_dir / "checkpoints" / "last.pt"), epoch, cfg,
                         extra={"val_cider": val_metrics.get("CIDEr"), "step": global_step})
            if epoch in save_epochs:
                import shutil
                shutil.copy(str(out_dir / "checkpoints" / "last.pt"),
                            str(out_dir / "checkpoints" / f"epoch_{epoch:02d}.pt"))
                run_progression_eval(epoch, val_metrics)

    print(f"[done] checkpoints in {out_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
