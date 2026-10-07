# CoCap Code Audit

Audit of the official implementation of *Accurate and Fast Compressed Video Captioning* (ICCV 2023, Shen et al.), cloned as the parent framework for the UAV Adaptive Captioning project.

**Audit date:** 2026-09-30
**Scope:** clone + full code audit + environment audit + baseline preparation. NO official code was modified. NO adaptive frame pruning was implemented (per project phase plan).

---

## 1. Official repository

| Item | Value |
|---|---|
| URL | https://github.com/yaojie-shen/CoCap |
| Paper | *Accurate and Fast Compressed Video Captioning*, ICCV 2023, arXiv:2309.12867 |
| Authors | Yaojie Shen, Xin Gu, Kai Xu, Heng Fan, Longyin Wen, Libo Zhang |
| License | MIT |
| Local clone | `/root/autodl-tmp/uav_adaptive_captioning/third_party/CoCap` (`.git` kept, 2.9 MB) |
| Working tree | clean, zero modifications |

**Important:** the `main` branch is a **2025 revised implementation** (Hydra + PyTorch Lightning rewrite of the original release). The paper's original implementation lives at tag `initial_release`. All findings below refer to `main` (the default branch, which the README documents).

## 2. Commit hash

```
FULL HASH:  828b0c128e9e32d12ad20a3b86a4ef4ac6fd4072
SHORT:      828b0c1
SUBJECT:    Minor updates
AUTHOR:     Yaojie Shen
DATE:       2025-07-28 13:11:13 +0800
TAG:        initial_release -> c42d593de3f29f7b2579e0daf1b70e490ecb9bcf (2023-09-25, original implementation)
BRANCHES:   main (HEAD), origin/dev == main at 828b0c1
REMOTE:     origin -> https://github.com/yaojie-shen/CoCap.git
```

## 3. Directory structure

```
CoCap/
├── README.md, LICENSE (MIT), pyproject.toml, requirements.txt, poster.pdf
├── assets/framework.svg                     # architecture figure
├── cocap/                                   # main package
│   ├── data/datasets/compressed_video/
│   │   ├── video_text_base.py               # get_video() dispatch + text tokenization helpers
│   │   ├── video_readers.py                 # ★ read_frames_compressed_domain() — H.264 bitstream reader
│   │   ├── transforms.py                    # Dict* transforms (crop/resize/flip/normalize on tensor dicts)
│   │   ├── compressed_video_utils.py        # JPEG (de)serialization for pre-extracted data
│   │   ├── dataset_msrvtt.py / dataset_msvd.py / dataset_vatex.py
│   ├── modeling/
│   │   ├── lm_cocap.py                      # ★ CoCapLM (PyTorch Lightning): train/val loops, decoding
│   │   ├── loss.py                          # LabelSmoothingLoss
│   │   ├── eval_captioning.py               # pycocoevalcap wrapper (Bleu/METEOR/ROUGE_L/CIDEr/SPICE)
│   │   └── optimization.py                  # BertAdam (legacy BERT-style Adam)
│   ├── modules/
│   │   ├── bert.py                          # ★ BertSelfEncoder (caption decoder), BertLMPredictionHead
│   │   ├── clip/                            # vendored OpenAI CLIP (model.py, clip.py, tokenizer)
│   │   └── compressed_video/
│   │       ├── compressed_video_transformer.py  # ★ CompressedVideoTransformer (visual encoder)
│   │       └── compressed_video_captioner.py    # ★ CompressedVideoCaptioner (top-level model)
│   └── utils/                               # checkpoint (legacy), train_utils, registry, logging, ...
├── configs/
│   ├── dataset/{msrvtt,msvd,vatex}.yaml     # per-dataset configs
│   └── exp/train/{base,msrvtt_captioning,msvd_captioning,vatex_captioning}.yaml
├── model_zoo/                               # CLIP pretrained weights only (urls.txt + download_model.sh)
├── dataset/README.md                        # data prep: H.264 re-encode (keyint=60, short edge 240)
├── tools/
│   ├── train_net.py                         # ★ training entry (hydra_zen)
│   ├── compute_cider.py                     # offline metric computation
│   ├── video_convert.py                     # parallel ffmpeg re-encode
│   ├── check_video_integrity.py, checkpoint_tweak.py, show_registry.py
└── test/                                    # unit tests incl. exact I/O shapes of the model
```

Dead code on `main`: `tools/show_registry.py` imports legacy modules (`cocap.config.base`, `cocap.data.build`, …) that no longer exist; `cocap/utils/checkpoint.py`, `train_utils.py`, `optimization.py` are leftovers of the pre-Lightning implementation (the current entry `tools/train_net.py` uses PL's own checkpointing).

## 4. Model architecture

Three-stage model, end-to-end on **compressed-domain** (H.264 bitstream) input — no pixel-space decoding of B/P frames:

1. **Visual encoder — `CompressedVideoTransformer`** (`cocap/modules/compressed_video/compressed_video_transformer.py`):
   - `rgb_encoder` = `IFrameEncoder` — CLIP ViT-B/16 initialized from pretrained weights, extended to also return hidden patch features. Encodes the I-frame of each GOP.
   - `motion_encoder` = CLIP-style `VisionTransformer` (from scratch, 2 layers / 8 heads, `in_channels=4`, patch 8, width 192). Encodes motion vectors (4-channel, H/4×W/4).
   - `residual_encoder` = CLIP-style `VisionTransformer` (from scratch, 2 layers / 8 heads, `in_channels=3`, patch 64, width 768). Encodes residuals (3-channel, full H×W).
   - `action_encoder` = `ActionEncoder` — 1 layer of cross-attention (`CrossResidualAttentionBlock`): each B/P-frame token (mv⊕res, summed CLS features) attends to its GOP's I-frame patch features, with learnable positional + B/P-type embeddings, then mean-pools over the GOP → one "action" feature per GOP.
2. **Caption decoder — `CaptionHead`** (`compressed_video_captioner.py`): a 2-layer causal BERT-style self-attention decoder (`BertSelfEncoder` in `modules/bert.py`) over the concatenation `[I-frame features | action features | text tokens]`, plus `BertLMPredictionHead` (weights initialized from the CLIP token embedding). Vocab = CLIP BPE (49,408).
3. **Top-level `CompressedVideoCaptioner`**: owns the transformer + caption head; applies motion/residual dropout; exposes the single `forward(inputs)` entry point.

Architecture figure: `assets/framework.svg`; per-module configs are built with `hydra_zen.builds(...)` next to each class.

## 5. Input format

`CompressedVideoCaptioner.forward` consumes one batch dict produced by the dataset + default collate:

```python
inputs = {
    "video": {
        "iframe":        (B, n_gop, 3, 224, 224)        # float, normalized (0.485/0.456/0.406)
        "motion_vector": (B, n_gop, n_mv, 4, 56, 56)   # float, H.264 MVs (4 ch), H/4 x W/4
        "residual":      (B, n_gop, n_res, 3, 224, 224)  # uint8 0..255 -> scaled to /128-1 in forward
        "type_ids_mv":   (B, n_gop, n_bp)              # long: 0=P, 1=B, 2=pad  (n_bp == n_mv == n_res)
        # input_mask_gop / input_mask_mv / input_mask_res: produced by reader, NOT consumed by model
    },
    "input_ids":   (B, 77)   # CLIP BPE token ids of the caption
    "input_labels":(B, 77)   # input_ids shifted left by 1 (teacher forcing target)
    "input_mask":  (B, 77)   # 1 = valid text token
}
```

Official config: `n_gop=8`, `n_mv=n_res=n_bp=59`, `max_t_len=77`, frames 224×224. Motion vectors must be H.264/AVC format (4-dim), asserted in `video_readers.py:248` and `compressed_video_transformer.py:199`.

Optional: if `inputs["visual_output"]` already contains pre-extracted visual features, the encoder is skipped entirely (`compressed_video_captioner.py:201-203`) — this cache path is used by the validation decoding loop.

## 6. Feature dimensions

With pretrained CLIP `ViT-B/16` (embed_dim=512, vision_width=768):

| Tensor | Shape | Producer |
|---|---|---|
| I-frame CLS per GOP — `feature_context` | (B, n_gop, **512**) | `IFrameEncoder` (CLIP ViT-B/16, 12L/12H, grid 14×14) |
| I-frame patch hidden — `f_ctx_all_hidden` | (B, n_gop, 196, **768**) | same encoder, `output_all_features=True` |
| Motion CLS per B/P frame — `mv_cls` | (B, n_gop, 59, **512**) | motion ViT (2L/8H, grid 7×7, width 192) |
| Residual CLS per B/P frame — `res_cls` | (B, n_gop, 59, **512**) | residual ViT (2L/8H, grid 3×3, width 768) |
| Fused action per GOP — `feature_action` | (B, n_gop, **512**) | `ActionEncoder` (1 cross-attn layer, mean-pool over 59) |
| Visual sequence into decoder | (B, **2·n_gop = 16**, 512) | concat[`feature_context`, `feature_action`] |
| Decoder output — `prediction_scores` | (B, 77, **49408**) | `BertLMPredictionHead` |

## 7. Frame / video processing

1. **Offline:** source videos are re-encoded to H.264 with `keyint=60`, short edge resized to 240 (`tools/video_convert.py`, `dataset/README.md`). The GOP structure is the core temporal unit.
2. **Bitstream parsing** — `read_frames_compressed_domain` (`video_readers.py:139`): `cv_reader.read_video()` (external Compressed-Video-Reader) returns per-frame `{pict_type, motion_vector, residual, rgb, frame_idx}`. Frames are grouped into GOPs starting at each I-frame; GOPs with ≤2 frames are dropped; GOPs are sampled (`resample_num_gop`), and per GOP `resample_num_mv`/`resample_num_res` B/P frames are sampled.
   - Sampling mode: `"rand"` in train, `"uniform"` in test — **but per-GOP B/P sampling is hard-coded `"rand"` even in test** (`video_readers.py:219-222`), a reproducibility subtlety.
   - `use_pre_extract=True` reads lz4+pickle per-video caches (`.pict_type`, `.rgb_gop`, `.motion_vector`, `.residual`) to skip bitstream parsing.
3. **Transforms** (`transforms.py`): dict-aware `CenterCrop(224)`, `RandomHorizontalFlip` (train), normalize (I-frames only, CLIP ImageNet stats). Motion vectors are cropped/resized at H/4×W/4 with NEAREST.
4. **Per-batch memory (fp32, B=2):** residual tensor alone ≈ 0.57 GB — comfortable on 24 GB.

## 8. Encoder

`CompressedVideoTransformer.forward(iframe, motion, residual, bp_type_ids)` (`compressed_video_transformer.py:179`):
- strict shape asserts (5D iframe / 6D motion+residual / 3D type ids, channel counts 3/4/3, H,W relationships) at lines 194–201;
- I-frames encoded per GOP in batch (`(B·n_gop, 3, H, W)`), motion/residual encoded per B/P frame in batch (`(B·n_gop·n_bp, C, H, W)`);
- fusion: `f_bp = mv_cls + res_cls` (line 240) → `ActionEncoder` cross-attends to the same GOP's I-frame patches, mean-pools → `f_act (B, n_gop, 512)`;
- returns `{feature_context, feature_action, iframe_attention_map, motion_vector_attention_map, residual_attention_map}`.
- Built via `CompressedVideoTransformer.from_pretrained(pretrained_clip_name_or_path="ViT-B/16", ...)`; CLIP JIT weights are loaded with `torch.jit.load` from `model_zoo/clip_model` (hard-coded download root in `CaptionHead.from_pretrained`).

## 9. Decoder

`CaptionHead` (`compressed_video_captioner.py:35`):
- `forward(visual_output, input_ids, input_mask)`: builds type ids (context=1, action=0, text=2), concatenates `[feature_context(8) | feature_action(8) | text(77)]` → sequence length 93, prepends visual ones to `input_mask`;
- `BertSelfEncoder` (`bert.py:226`): learnable projections (visual 512→512, word 512→512), sinusoidal position encodings (max_len 1000), 3-way token-type embeddings, 2 layers of `BertLayerNoMemory` with a **causal shifted mask** (`make_shifted_mask`, `bert.py:12`): video tokens visible to everyone; text tokens attend to all video + preceding text only;
- `BertLMPredictionHead`: dense+gelu+LN transform → linear decoder (weights tied to the CLIP word embedding, loaded in `from_pretrained`) + bias → `(B, 77, 49408)`;
- **Decoding is greedy argmax only** — there is no beam search anywhere in the repo (verified by grep). At inference, one autoregressive step per token (max 77), with visual features computed once and cached via the `visual_output` bypass.

## 10. Training pipeline

- **Entry:** `python3 tools/train_net.py --config-name=exp/train/msrvtt_captioning` — hydra_zen + PL (`tools/train_net.py:29-52`).
- **Module:** `CoCapLM` (`modeling/lm_cocap.py:45`) — `training_step` = `model(batch)` + `LabelSmoothingLoss`; teacher forcing with `input_labels = input_ids shifted`.
- **Loss:** label smoothing 0.1, vocab 49,408, KL-div, PAD id 0 ignored (`modeling/loss.py:30`).
- **Optimizer:** custom `BertAdam` with weight-decay split (pretrained CLIP params → lr 1e-6, no decay; others → lr 1e-4), warmup 5% of total steps, ×0.95 decay per epoch (`lm_cocap.py:75-168`).
- **Config defaults** (`configs/exp/train/base.yaml`): 20 epochs, batch 2/GPU, grad-accum 4, `ddp_find_unused_parameters_true` (works single-GPU), ModelCheckpoint `save_top_k=-1`, TensorBoard, fork multiprocessing context, num_workers 4.

## 11. Inference pipeline

- No standalone inference script. Caption generation happens in `CoCapLM.validation_step` (`lm_cocap.py:182-212`): zero the text mask, greedy-decode 77 steps (one `model()` call per step; the encoder runs only on step 0, its output is cached into `batch["visual_output"]`), detokenize with `convert_ids_to_sentence`.
- Predictions are saved as JSON (`caption_greedy_pred_validation_<ts>.json`) in the log dir and metrics are computed in `on_validation_epoch_end`.

## 12. Evaluation metrics

`EvalCap` / `evaluate` (`modeling/eval_captioning.py`): pycocoevalcap scorers — **Bleu_1-4, METEOR, ROUGE_L, CIDEr** (SPICE implemented but disabled by default). METEOR requires Java (present: OpenJDK 11.0.32). MSRVTT test reference = 20 captions/video (asserted in `dataset_msrvtt.py:88`). `tools/compute_cider.py` recomputes metrics offline from saved JSON.

## 13. Dataset support

- **MSRVTT / MSVD / VATEX** captioning datasets, each reading H.264-compressed videos through the same reader + config (`configs/dataset/*.yaml`, all `n_gop=8, n_mv=59, n_res=59, with_residual=True`).
- MSRVTT splits are hard-coded in code: first 6513 train / next 497 val / rest test of `MSRVTT_data.json` (`dataset_msrvtt.py:47-48`). MSVD/VATEX take splits from the metadata JSON.
- Legacy raw-frame readers (`read_frames_cv2/av/decord`) are registered but the captioning datasets hard-wire `read_frames_compressed_domain`.
- Annotations tarball is hosted on **Google Drive** (unreachable from this host — see memory on AutoDL network facts); videos from dataset homepages; `video_convert.py` re-encodes them.

## 14. Checkpoint availability

- **No trained CoCap checkpoints are released** — `model_zoo/` contains only CLIP pretrained weights (ViT-B-16.pt etc., ~350 MB, from openaipublic.azureedge.net). A baseline must be trained from scratch.
- Training checkpoints are saved by PL `ModelCheckpoint` (full LightningModule incl. optimizer state, all epochs). `cocap/utils/checkpoint.py` is legacy and unused by the main-branch entry; `tools/checkpoint_tweak.py` can extract/prune state dicts.

## 15. Exact insertion point for Adaptive Frame Pruning

**The single interface where video data enters the captioning model is `CompressedVideoCaptioner.forward(inputs)` (`cocap/modules/compressed_video/compressed_video_captioner.py:175`).**

```text
Original H.264 video
   ↓  (offline) tools/video_convert.py — keyint=60, short edge 240
dataset/<name>/videos_h264_keyint_60/*.mp4
   ↓
cocap/data/datasets/compressed_video/video_readers.py
   read_frames_compressed_domain(video_path, resample_num_gop=8, resample_num_mv=59, ...)
   → video dict {iframe (8,3,224,224), motion_vector (8,59,4,56,56),
                 residual (8,59,3,224,224), type_ids_mv (8,59), masks}
   ↓  collate → batch tensors get dim B
cocap/modules/compressed_video/compressed_video_captioner.py
   CompressedVideoCaptioner.forward(inputs)          ★ THE INTERFACE
   inputs["video"]: iframe (B,8,3,224,224) · motion (B,8,59,4,56,56)
                    · residual (B,8,59,3,224,224) · type_ids_mv (B,8,59)
                    · input_ids/input_mask (B,77)
   ↓
cocap/modules/compressed_video/compressed_video_transformer.py
   CompressedVideoTransformer.forward(iframe, motion, residual, bp_type_ids)
     ├─ IFrameEncoder (CLIP ViT-B/16)   → f_ctx_cls (B,8,512) + patches (B,8,196,768)
     ├─ motion VisionTransformer       → mv_cls (B,8,59,512)
     ├─ residual VisionTransformer     → res_cls (B,8,59,512)
     └─ ActionEncoder (cross-attn + mean-pool) → f_act (B,8,512)
   → visual_output {"feature_context": (B,8,512), "feature_action": (B,8,512), attn maps}
   ↓
compressed_video_captioner.py — CaptionHead.forward(visual_output, input_ids, input_mask)
   concat[feature_context(8) | feature_action(8) | text(77)] → (B, 93, 512)
   → BertSelfEncoder (2-layer causal) → BertLMPredictionHead → (B, 77, 49408)
```

## 16. Pruning feasibility analysis

**1. Can pruning be inserted before the CoCap encoder? Yes — cleanly, at two levels:**

- **Level A (recommended) — GOP-level, before `compressed_video_transformer`:** a selector module (project-side) takes the `video` dict, picks `k` of the `n_gop` GOPs, and slices **all four tensors jointly** (`iframe`, `motion_vector`, `residual`, `type_ids_mv` — plus masks if ever used) along the GOP axis. This saves encoder compute (I-frame ViT-B/16 is the dominant cost), which is exactly the project's goal of reducing visual computation.
- **Level B — feature-level, between transformer and caption head:** pick `k` rows of `feature_context`/`feature_action`. Only saves decoder compute; not the primary target.

**2. Raw frames vs features?** Prune on **raw compressed tensors (GOP level)**, not on encoded features: (i) it achieves the compute saving; (ii) CoCap's natural temporal unit is the **GOP**, not the single frame — the three streams are coupled per GOP and must be pruned together; (iii) per-frame pruning inside a GOP (rows of `n_mv`) is possible but the ActionEncoder's mean-pool over 59 tokens has no mask support, so partially pruned GOPs would dilute features — GOP-granularity is the clean unit.

**3. Feature dimensions if pruning on features:** `feature_context` (B, n_gop, 512), `feature_action` (B, n_gop, 512); embed_dim = **512** (ViT-B/16).

**4. Does it break motion/residual logic?** No, if GOP-granular and joint. One constraint: `CaptionHead` config has fixed `max_v_len = 2·n_gop = 16`, used by the causal mask in `BertSelfEncoder.forward` (`bert.py:269`). For a pruned count `k`: since `cap_config` is a shared `EasyDict` object mutated at runtime (`caption_head.cap_config.max_v_len = 2*k`), **variable-length pruning requires zero official-code changes** — the decoder accepts any visual length (position encodings are sinusoidal, type embeddings are length-independent). For batched training, use a **fixed budget k** per batch (tensor batching requires uniform GOP counts); variable k is fine at inference (batch size 1) or with bucketing.
   Also note: to give the selector a real candidate pool, raise `cv_config.num_gop` (e.g., 8→16) in the dataset YAML — the reader already parameterizes it, no code change needed.

**5. Minimal file set for later modification:**
- **New project-side files** (in `models/` of the UAV project, not in third_party): the adaptive selector module + its hydra config (candidates → scores → budget → keep mask).
- **Zero official files required** if implemented as a project-side subclass of `CompressedVideoCaptioner` overriding `forward` (insert selector, then call official parent modules), plus optional runtime mutation of `cap_config.max_v_len`.
- Optionally touch later (project-side only): a `CoCapLM` subclass in `cocap/modeling/lm_cocap.py` style if a budget regularizer must join the captioning loss; dataset YAML for the candidate-pool size.

**6. Files that must remain untouched:** everything under `third_party/CoCap/` — in particular `compressed_video_captioner.py`, `compressed_video_transformer.py`, `bert.py`, `clip/*`, `video_readers.py`, `dataset_*.py`, `lm_cocap.py`, `eval_captioning.py`, `configs/`. All integration happens via subclassing/wrapping from project code.

## 17. Environment / dependency issues

Current env `uavcap` (Python 3.10.13, torch 2.1.2+cu121, torchvision 0.16.2, numpy 1.26.4, RTX 4090 24 GB) vs `requirements.txt`:

**No torch/numpy/torchvision upgrade needed.** CoCap pins torch==2.5.1, but the code uses no 2.5-specific API; PyTorch Lightning 2.5.1 only requires torch>=1.13. torch 2.1.2 is compatible.

| Package | Status | Action |
|---|---|---|
| torch / torchvision / numpy | OK (2.1.2 / 0.16.2 / 1.26.4) | **do not touch** |
| omegaconf 2.3.1, fvcore, einops, cv2, PIL, pandas, tqdm, joblib, tabulate, h5py, pyyaml, pycocoevalcap, timm 0.9.16 | OK | none (timm is pinned 1.0.15 but **not imported** by main-branch code) |
| pytorch-lightning, hydra-core, hydra-zen, easydict, decord, lz4, ftfy, colorlog, flow_vis, terminaltables, av (optional) | **missing** | `pip install` into uavcap (new packages only, no version bumps of existing) |
| **cv_reader** (Compressed-Video-Reader) | **missing** | build from https://github.com/yaojie-shen/Compressed-Video-Reader: `install.sh` downloads, patches, and compiles FFmpeg + a pybind11 extension. Build tools present (gcc 11.4, cmake 3.22); expect 15–30 min build |
| ffmpeg/ffprobe binaries | **missing** (libavcodec 58 dev headers + runtime libs present) | needed by `tools/video_convert.py` and `/usr/bin/ffprobe` in `get_frame_type`; `apt install ffmpeg` or conda-forge ffmpeg |
| Java | OK (OpenJDK 11.0.32) | required by METEOR — done |
| CLIP ViT-B-16.pt (~350 MB) | not downloaded | `model_zoo/download_model.sh` (aria2) or HF mirror; hard-coded path `model_zoo/clip_model` |

First import failure today: `ModuleNotFoundError: No module named 'easydict'` (then hydra_zen, pytorch_lightning, decord, cv_reader in order).

## 18. Baseline reproduction status

**Cannot run yet.** Blockers, in fix order:
1. Missing Python deps (PL, hydra_zen, easydict, decord, lz4, …) — pure-pip fix, no risk to torch/numpy.
2. `cv_reader` not built — heavyweight but self-contained build (patched FFmpeg + extension).
3. No ffmpeg binary on the host — needed for data prep and `ffprobe`.
4. CLIP ViT-B-16.pt not downloaded (network: official CDN unverified from this host; HF mirror available via academic accelerator).
5. **No released CoCap checkpoints** — baseline must be trained from scratch on MSRVTT (20 epochs, batch 2 × grad-accum 4).
6. Data not downloaded: annotations tarball on Google Drive (unreachable from this host → user download/upload, same flow as CapERA); MSRVTT videos from official mirrors.

Validation path after env fix: official unit tests (`test/`) confirm exact model I/O shapes; `test_compressed_video_captioner.py` and `test_compressed_video_transformer.py` only need the CLIP weights; `test_video_readers.py` needs a real H.264 video.

## 19. Summary for the Adaptive Frame Pruning plan

- **Interface:** `CompressedVideoCaptioner.forward` (`compressed_video_captioner.py:175`), inputs = `video` dict of GOP-structured compressed tensors + text.
- **Pruning unit:** GOP (I-frame + 59 B/P frames), jointly across all streams.
- **Insertion:** Level A (before `compressed_video_transformer`) via a project-side subclass/wrapper — official code stays untouched; candidate pool raised via dataset YAML only; `cap_config.max_v_len` mutation enables variable budgets with zero official changes.
- **Next phases (not started, per plan):** baseline training first, then the selector (candidates → scoring → budget → keep-mask) with temporal context.
