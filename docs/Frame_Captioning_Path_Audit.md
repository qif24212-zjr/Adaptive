# Frame Captioning Path Audit

**Date:** 2026-09-30
**Goal:** extend CoCap (compressed-domain video captioning) with an **RGB frame-level input path** that accepts a variable number of frames `N` and feeds their features into the official CoCap caption decoder — WITHOUT implementing any frame-pruning algorithm and WITHOUT modifying official code.

All findings below are from executed code (smoke tests were run for real on the RTX 4090).

---

## 1. CoCap original data flow (verified from code)

```text
H.264 video (keyint=60)
  -> cocap/data/datasets/compressed_video/video_readers.py
       read_frames_compressed_domain()  (cv_reader bitstream parsing, GOP sampling)
  -> video dict: iframe (B,8,3,224,224) · motion_vector (B,8,59,4,56,56)
                 residual (B,8,59,3,224,224) · type_ids_mv (B,8,59)
  -> cocap/modules/compressed_video/compressed_video_captioner.py
       CompressedVideoCaptioner.forward(inputs)          [L175]
       (residual/128-1, motion/residual dropout 0.2)
  -> compressed_video_transformer.py
       CompressedVideoTransformer.forward()              [L179]
         IFrameEncoder (CLIP ViT-B/16)  -> feature_context  (B,8,512)
         motion/residual ViTs + ActionEncoder -> feature_action (B,8,512)
  -> CaptionHead.forward(visual_output, input_ids, input_mask)   [L95]
       concat[feature_context(8,type=1) | feature_action(8,type=0) | text(77,type=2)]
       -> BertSelfEncoder (2-layer causal) -> BertLMPredictionHead
       -> prediction_scores (B,77,49408)
```

**Key conclusion (the crux of the refactor):** `CaptionHead` does NOT know anything about I-frames / motion vectors / residuals. It consumes only two already-encoded visual streams of shape `(B, T, 512)` plus text tokens. The entire compressed-domain coupling lives in `CompressedVideoTransformer` and the `video` dict.

## 2. New frame-level data flow (implemented, all smoke-tested)

```text
RGB video (any format cv2 can decode)
  -> models/frame_cocap/frame_sampling.py
       uniform_sample_indices() + read_video_frames_cv2()   (official sampling math,
       center-crop 224, ImageNet normalization)             N frames, N variable
  -> frames (B, N, 3, 224, 224)         (batch: pad to N_max via collate_frames + mask)
  -> models/frame_cocap/frame_video_captioner.py
       FrameVideoCaptioner.forward(frames, frame_mask, input_ids, input_mask)
         -> FrameEncoder (per-frame, [B*N, 3, 224, 224])
         -> visual_features (B, N, 512)
         -> FrameCaptionHead (single visual stream + visual mask, type id = 1)
         -> prediction_scores (B, 77, 49408)
```

`FrameCaptionHead` subclasses the official `CaptionHead`. Only two official behaviors are overridden: (a) one visual stream instead of two; (b) `cap_config.max_v_len` is set per forward — the config is a shared EasyDict with `BertSelfEncoder`, so variable N works with zero official changes. `BertSelfEncoder`, `BertLMPredictionHead`, tied CLIP word embeddings, tokenizer, vocabulary (49,408), causal mask, greedy decoding, loss interface: all official, all unchanged.

## 3. Files: added / modified / untouched

**Added (project-side only):**
| File | Role |
|---|---|
| `models/frame_cocap/__init__.py` | package exports |
| `models/frame_cocap/frame_video_captioner.py` | `FrameCaptionHead`, `CLIPFrameEncoder`, `MaxViTFrameEncoder`, `FrameVideoCaptioner`, `build_frame_captioner`, `tokenize_captions`, `decode_ids`, `generate_caption` |
| `models/frame_cocap/frame_sampling.py` | `uniform_sample_indices`, `collate_frames`, `read_video_frames_cv2` |
| `configs/frame_captioning.yaml` | `num_frames: 8`, `frame_sampling: uniform`, `frame_encoder: clip_vitb16`, `feature_dim: 512`, ... |
| `scripts/smoke_test_frame_cocap.py` | executed smoke-test suite |
| `scripts/run_uniform8_baseline.py` | uniform-8 baseline on real CapERA videos |

**Modified:** nothing in `third_party/CoCap/` — `git status` shows **0 dirty files**.

**Environment (additive only):** installed `easydict`, `hydra-core==1.3.2`, `hydra-zen==0.15.0`, `ftfy`, `wcwidth` (pure Python, required by official cocap imports); downloaded `checkpoints/clip/ViT-B-16.pt` (350 MB, sha256 verified) to the project checkpoints dir (NOT into the official repo, keeping its git tree clean). torch / torchvision / numpy untouched.

**Untouched:** all official files (`compressed_video_captioner.py`, `compressed_video_transformer.py`, `bert.py`, `clip/*`, `video_readers.py`, datasets, `lm_cocap.py`, `eval_captioning.py`, configs). The original compressed-domain path still builds and runs (verified in the smoke test: `CompressedVideoTransformer.from_pretrained` + forward on (2,8,59) compressed tensors → `feature_context (2,8,512)`, `feature_action (2,8,512)`).

## 4. Tensor shapes (measured, not estimated)

```text
RGB frames                (B, N, 3, 224, 224)    float, ImageNet-normalized   N variable
collated (padded) frames  (B, N_max, 3, 224, 224)  + frame_mask (B, N_max) long
frame encoder input       (B*N, 3, 224, 224)     (per-frame batching)
frame features            (B, N, 512)            CLIP ViT-B/16 CLS (official IFrameEncoder)
                                                 MaxViT-S: (B, N, 768) -> proj -> 512
caption decoder input     (B, N + 77, 512)       [frame tokens(type 1) | text(type 2)]
decoder hidden            (B, N + 77, 512)       2-layer causal BERT (official BertSelfEncoder)
logits                    (B, 77, 49408)         official BertLMPredictionHead
```

## 5. Variable-N support — all verified by execution

| Config | Result |
|---|---|
| B=1, N=4 | ✓ forward (1,77,49408) |
| B=1, N=8 | ✓ |
| B=1, N=12 | ✓ |
| B=1, N=16 | ✓ |
| Mixed batch N=[4,8,12,16] → pad to 16 + mask | ✓ forward (4,77,49408) |
| **Mask correctness** (padding slots filled with ×100 noise vs zeros) | ✓ **max |Δlogits| = 0.0e+00** — padded visual tokens are fully masked from the decoder |
| Greedy generation (synthetic frames, N=4/8) | ✓ runs end-to-end |

Mechanism: within a batch, pad frames to N_max and pass `frame_mask`; `FrameCaptionHead` concatenates it with the text mask, so the official `make_pad_shifted_mask` masks padded visual tokens as attention keys. The decoder never sees them.

## 6. Smoke test — actually executed

`python scripts/smoke_test_frame_cocap.py` → **ALL TESTS PASSED** (GPU, RTX 4090):
1. B=1, N ∈ {4, 8, 12, 16} forward ✓
2. variable-length batch [4, 8, 12, 16] + padding/mask ✓
3. mask correctness (padding-invariance, exact 0 diff) ✓
4. greedy caption generation ✓
5. official compressed-domain path: git tree clean (0 dirty files) + `CompressedVideoTransformer` still runs ✓

## 7. Uniform-8 baseline (real data inference)

Real CapERA test videos were extracted from the official `ERA_Dataset.zip` (1391 mp4s under `datasets/CapERA/videos/Videos/Test/`, originals preserved; note the zip ships each video twice, as `X .mp4` and `X.mp4` — the script indexes both spellings).

`python scripts/run_uniform8_baseline.py` → video → uniform 8 frames (cv2) → CLIP ViT-B/16 → FrameCaptionHead (greedy) → caption, saved to `experiments/uniform8_framecocap/pred_test_n8.json`, evaluated with the official pycocoevalcap wrapper (Bleu_1-4 / METEOR / ROUGE_L / CIDEr vs 5 refs).

**Status:** full-test-set run executed on 2026-09-30 — **1391/1391 videos, 0 skipped, 612.6 s**, predictions at `experiments/uniform8_framecocap/pred_test_n8.json`. Metrics (pycocoevalcap, 5 refs): Bleu_1-4 0.00 / METEOR 0.11 / ROUGE_L 0.00 / CIDEr 0.00; captions degenerate (median length 1 token). Expected by design: the caption head is only *pretrained-initialized* (CLIP token embeddings), **no captioning training has happened yet**. This run is path verification, not model quality. Captioning training (CapERA or MSRVTT) is the next phase, before the selector.

## 8. Frame encoder choice (Option A vs Option B)

| | Option A: CLIP ViT-B/16 | Option B: MaxViT-S |
|---|---|---|
| Source | official CoCap `IFrameEncoder` (pretrained CLIP JIT, sha256-verified local file) | project CapERA baseline backbone (timm, pretrained) |
| Output dim | **512 native** = CoCap embed_dim | 768 → needs learned 512-projection |
| Feature space | identical to CoCap's `feature_context` (same backbone the compressed path uses for I-frames) | different space; the CapERA baseline's space |
| Integration cost | zero projection, direct | one linear layer |
| Verdict | **primary** (faithful to CoCap, zero adapters) | supported as alternative (`frame_encoder: maxvit_s` in config) |

## 9. Compatibility summary

- The official compressed-domain path and the new frame-level path **coexist**; either can be trained/evaluated independently → direct comparison of "CoCap original vs CoCap + Adaptive Frame Pruning" is possible.
- Decoder-side everything (vocab, tied embeddings, causal mask, teacher forcing, loss, greedy decode, pycocoevalcap eval) is reused **unchanged** from official code.
- The only model-side constraint: visual features must be 512-d (projected if the encoder is not CLIP ViT-B/16).
- Next phase plugs in cleanly: `Candidate Frames → [my selector] → Selected Frames → frame_cocap.FrameVideoCaptioner → caption`. The selector only needs to produce `frames`/`frame_mask` (or `visual_features`/`frame_mask`).

## 10. Known limitations (honest)

1. Caption head is untrained for captioning — baseline metrics are path-sanity only; training is the next phase.
2. Padding slots are still *encoded* by the frame encoder (compute not saved for padded frames). A future efficiency pass can encode only valid frames per sample and scatter the results.
3. Text tokens' absolute positions shift with N_max across batches — harmless because CoCap's position encodings are fixed sinusoidal (no learned positional params).
4. The frame path has no inter-frame temporal fusion (frames are independent decoder tokens, exactly like CoCap's `feature_context`). Temporal context is exactly what the future Adaptive Frame Pruning phase will add upstream of this interface.
