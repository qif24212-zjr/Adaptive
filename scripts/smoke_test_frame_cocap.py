# -*- coding: utf-8 -*-
"""
Smoke test for the CoCap frame-level captioning path.

Verifies (with REAL forward passes on GPU):
  1. RGB frames -> FrameEncoder -> [B,N,512] -> FrameCaptionHead -> logits
     for B=1, N in {4, 8, 12, 16}
  2. Variable-length batch: N = [4, 8, 12, 16] padded to 16 with mask
  3. Mask correctness: padding VALUES must not affect text logits
     (padded visual tokens are masked as attention keys)
  4. Greedy caption generation end-to-end (synthetic frames)
  5. Official compressed-domain model still builds & runs (untouched)

Run:  python scripts/smoke_test_frame_cocap.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.frame_cocap import (
    build_frame_captioner,
    collate_frames,
    generate_caption,
    tokenize_captions,
)

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_frames(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, 224, 224, generator=g)  # synthetic RGB (already normalized)


def main():
    print(f"device: {DEVICE}")
    print("[0] building FrameVideoCaptioner (CLIP ViT-B/16 encoder + pretrained CaptionHead) ...")
    model = build_frame_captioner(
        frame_encoder="clip_vitb16",
        clip_path="checkpoints/clip/ViT-B-16.pt",
    ).to(DEVICE).eval()
    print(f"    frame encoder: {type(model.frame_encoder).__name__} "
          f"(feature_dim={model.frame_encoder.feature_dim})")
    print(f"    caption head:  {type(model.caption_head).__name__} "
          f"(vocab={model.caption_head.cap_config.vocab_size}, "
          f"BOS={model.caption_head.cap_config.BOS_id}, EOS={model.caption_head.cap_config.EOS_id})")
    print(f"    tied embeddings (values): "
          f"{torch.equal(model.caption_head.cap_sa_decoder.word_embeddings.weight, model.caption_head.prediction_head.decoder.weight)}")

    print("\n[1] B=1, N in {4, 8, 12, 16} -> forward")
    captions = ["a drone flies over a river"]
    for n in (4, 8, 12, 16):
        frames = make_frames(n).unsqueeze(0).to(DEVICE)  # (1, N, 3, 224, 224)
        with torch.no_grad():
            logits = model(frames=frames, captions=captions)
        assert logits.shape == (1, 77, 49408), f"N={n}: {logits.shape}"
        print(f"    N={n:2d}  frames {tuple(frames.shape)} -> logits {tuple(logits.shape)}  OK")

    print("\n[2] variable-length batch: N = [4, 8, 12, 16] -> pad + mask")
    frames_list = [make_frames(n, seed=i) for i, n in enumerate((4, 8, 12, 16))]
    frames, mask = collate_frames([f.to(DEVICE) for f in frames_list])
    print(f"    collated frames {tuple(frames.shape)}, mask {tuple(mask.shape)}")
    print(f"    mask rows: {mask.sum(1).tolist()}")
    captions = ["a drone flies over a river"] * 4
    with torch.no_grad():
        logits = model(frames=frames, frame_mask=mask, captions=captions)
    assert logits.shape == (4, 77, 49408)
    print(f"    padded batch forward -> logits {tuple(logits.shape)}  OK")

    print("\n[3] mask correctness: padding values must not change logits")
    frames_noise = frames.clone()
    # fill padded slots (per-row) with wildly different values
    for i, n in enumerate((4, 8, 12, 16)):
        frames_noise[i, n:] = torch.randn_like(frames_noise[i, n:]) * 100
    with torch.no_grad():
        logits_zero = model(frames=frames, frame_mask=mask, captions=captions)
        logits_noise = model(frames=frames_noise, frame_mask=mask, captions=captions)
    max_diff = (logits_zero - logits_noise).abs().max().item()
    print(f"    max |logits(zero-pad) - logits(noise-pad)| = {max_diff:.3e}")
    assert max_diff < 1e-4, "padded visual tokens leak into the decoder!"
    print("    padded tokens are fully masked from the caption decoder  OK")

    print("\n[4] greedy caption generation (synthetic frames)")
    for n in (4, 8):
        frames = make_frames(n).unsqueeze(0).to(DEVICE)
        cap = generate_caption(model, frames=frames)[0]
        print(f"    N={n}: '{cap}'")

    print("\n[5] official compressed-domain path untouched (git clean + still runs)")
    import subprocess
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd="third_party/CoCap",
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"    official git tree dirty files: {len(git_status.splitlines())}")
    assert git_status == "", "official CoCap tree was modified!"
    from cocap.modules.compressed_video import CompressedVideoTransformer
    official_enc = CompressedVideoTransformer.from_pretrained(
        pretrained_clip_name_or_path="checkpoints/clip/ViT-B-16.pt",
    ).to(DEVICE).eval()
    with torch.no_grad():
        out = official_enc(
            iframe=torch.rand(2, 8, 3, 224, 224, device=DEVICE),
            motion=torch.rand(2, 8, 59, 4, 56, 56, device=DEVICE),
            residual=torch.rand(2, 8, 59, 3, 224, 224, device=DEVICE),
            bp_type_ids=torch.randint(0, 2, (2, 8, 59), device=DEVICE),
        )
    print(f"    official CompressedVideoTransformer forward -> "
          f"feature_context {tuple(out['feature_context'].shape)}, "
          f"feature_action {tuple(out['feature_action'].shape)}  OK")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
