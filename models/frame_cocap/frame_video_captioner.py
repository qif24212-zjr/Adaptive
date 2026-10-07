# -*- coding: utf-8 -*-
"""
Frame-level captioning path for CoCap.

RGB frames -> frame encoder -> [B, N, D] visual features -> CoCap CaptionHead
-> [B, L, V] logits.

N (number of frames) is VARIABLE: within a batch, pad to N_max and pass
`frame_mask` so the caption decoder ignores padded visual tokens.

Official CoCap code (third_party/CoCap) is NOT modified. Everything here
subclasses/composes official classes; the official compressed-domain path
(CompressedVideoCaptioner / CompressedVideoTransformer) stays fully intact.

Data flow:

    frames [B, N, 3, 224, 224]        (N variable; pad to N_max in a batch)
        -> FrameEncoder (per-frame, batches as [B*N, 3, 224, 224])
        -> visual_features [B, N, D]  (D = 512 = CoCap embed_dim)
        -> FrameCaptionHead (single visual stream + visual mask)
        -> prediction_scores [B, 77, 49408]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

# Make the official CoCap package importable WITHOUT pip-installing it
# (pip install would pull its full dependency set incl. torch pins; the
# sys.path trick keeps the official tree read-only).
_COCAP_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "CoCap"
if str(_COCAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_COCAP_ROOT))

from cocap.modules.clip import clip  # noqa: E402  (CLIP tokenizer)
from cocap.modules.compressed_video.compressed_video_captioner import CaptionHead  # noqa: E402
from cocap.modules.compressed_video.compressed_video_transformer import IFrameEncoder  # noqa: E402

__all__ = [
    "FrameCaptionHead",
    "CLIPFrameEncoder",
    "MaxViTFrameEncoder",
    "FrameVideoCaptioner",
    "build_frame_captioner",
    "tokenize_captions",
    "decode_ids",
    "generate_caption",
]

# CLIP BPE vocab: PAD=0, BOS=vocab-2=49406, EOS=vocab-1=49407
MAX_T_LEN = 77
VOCAB_SIZE = 49408


class FrameCaptionHead(CaptionHead):
    """CaptionHead variant taking ONE visual stream of frame features.

    The official CaptionHead expects exactly two streams
    (feature_context with type id 1, feature_action with type id 0, each
    [B, T, D]) and a fixed max_v_len = 2*T. The frame path has a single
    stream [B, N, D] (type id 1, "content/context" semantics), with N
    variable per sample (padded to N_max within a batch).

    Differences vs official forward():
      1. single visual stream + explicit visual_mask (B, N);
      2. cap_config.max_v_len is set per forward. cap_config is the SAME
         EasyDict shared with BertSelfEncoder (official code passes it by
         reference), so the causal attention mask picks up the new length
         with zero changes to official code.

    Everything else - BertSelfEncoder, BertLMPredictionHead, tied CLIP word
    embeddings, label-smoothing loss, teacher forcing - is reused as-is.
    """

    def forward(
            self,
            visual_features: Tensor,   # (B, N, D)
            visual_mask: Tensor,       # (B, N) long, 1 = valid frame token
            input_ids: Tensor,         # (B, max_t_len) long
            input_mask: Tensor,        # (B, max_t_len) long, 1 = valid text token
    ) -> Tensor:
        assert input_ids.size(1) == self.cap_config.max_t_len, \
            f"{input_ids.size(1)} vs {self.cap_config.max_t_len}"
        B, N, D = visual_features.shape

        # variable-length support: the shared config drives the causal mask
        # inside BertSelfEncoder (official code reads config.max_v_len)
        self.cap_config.max_v_len = N

        input_types = torch.cat(
            [
                torch.full((B, N), fill_value=1, dtype=torch.long, device=visual_features.device),
                torch.full((B, input_ids.size(1)), fill_value=2, dtype=torch.long,
                           device=input_ids.device),
            ], dim=1
        )  # (B, N + max_t_len)
        input_mask = torch.cat([visual_mask, input_mask], dim=1)  # (B, N + max_t_len)

        hidden = self.cap_sa_decoder.forward(visual_features, input_ids, input_mask, input_types)
        prediction_scores = self.prediction_head(hidden[:, -self.cap_config.max_t_len:])
        return prediction_scores  # (B, max_t_len, vocab)


class CLIPFrameEncoder(nn.Module):
    """Per-frame encoder: the official CoCap IFrameEncoder (CLIP ViT-B/16).

    This is the SAME visual backbone CoCap uses for I-frames in the
    compressed-domain path, so the frame features live in the identical
    512-d space as CoCap's feature_context. Output dim == CoCap embed_dim
    natively, no projection needed.
    """

    def __init__(self, clip_path: Union[str, Path]):
        super().__init__()
        # official classmethod: loads pretrained CLIP JIT weights
        # returns (encoder, image_resolution, vision_width, embed_dim)
        self.encoder, self.image_resolution, self.vision_width, self.embed_dim = \
            IFrameEncoder.from_pretrained(str(clip_path))
        self._feature_dim = self.embed_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, frames: Tensor) -> Tensor:
        """frames: (B*N, 3, H, W) -> (B*N, 512) CLS features."""
        # official IFrameEncoder returns a tuple (cls_feature,) by default
        return self.encoder(frames)[0]


class MaxViTFrameEncoder(nn.Module):
    """Per-frame encoder: timm MaxViT-S (768-d) + learned projection -> 512-d.

    Option B: reuse the project's existing MaxViT-S backbone (the CapERA
    baseline encoder). 768-d features are projected to the 512-d space
    expected by CoCap's CaptionHead. Also supports feeding PRE-EXTRACTED
    768-d features (features_only mode) for cheap inference.
    """

    def __init__(self, feature_dim: int = 512, pretrained: bool = True):
        super().__init__()
        import timm
        self.backbone = timm.create_model("maxvit_small_tf_224", pretrained=pretrained, num_classes=0)
        self.backbone_dim = 768
        self.proj = nn.Linear(self.backbone_dim, feature_dim)
        self._feature_dim = feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, frames: Optional[Tensor] = None, features: Optional[Tensor] = None) -> Tensor:
        """Either encode raw frames (B*N, 3, H, W) or project precomputed
        768-d features (B*N, 768) -> (B*N, 512)."""
        if features is None:
            assert frames is not None, "provide frames or precomputed features"
            features = self.backbone.forward_features(frames)
        return self.proj(features)


class FrameVideoCaptioner(nn.Module):
    """RGB frame path: frames -> frame_encoder -> [B, N, D] -> FrameCaptionHead -> logits.

    Usage:
        outputs = model(frames=frames, captions=captions)          # raw strings
        outputs = model(frames=frames, input_ids=ids, input_mask=m)
        outputs = model(visual_features=f, frame_mask=m, input_ids=ids, input_mask=m)
    """

    def __init__(self, frame_encoder: nn.Module, caption_head: FrameCaptionHead):
        super().__init__()
        self.frame_encoder = frame_encoder
        self.caption_head = caption_head

    def forward(
            self,
            frames: Optional[Tensor] = None,           # (B, N, 3, H, W), N variable
            captions: Optional[List[str]] = None,      # raw caption strings
            frame_mask: Optional[Tensor] = None,       # (B, N) long, 1 = valid
            input_ids: Optional[Tensor] = None,        # (B, 77) long
            input_mask: Optional[Tensor] = None,       # (B, 77) long
            visual_features: Optional[Tensor] = None,  # (B, N, D) precomputed
    ) -> Tensor:
        if visual_features is None:
            assert frames is not None, "provide frames or visual_features"
            B, N, C, H, W = frames.shape
            feats = self.frame_encoder(frames.reshape(B * N, C, H, W))
            visual_features = feats.reshape(B, N, -1)
        else:
            B, N, D = visual_features.shape

        if frame_mask is None:
            frame_mask = torch.ones((B, N), dtype=torch.long, device=visual_features.device)

        if captions is not None:
            input_ids, input_mask = tokenize_captions(captions)
        assert input_ids is not None and input_mask is not None, \
            "provide captions or tokenized input_ids/input_mask"
        input_ids = input_ids.to(device=visual_features.device)
        input_mask = input_mask.to(device=visual_features.device)

        prediction_scores = self.caption_head(visual_features, frame_mask, input_ids, input_mask)
        return prediction_scores  # (B, 77, 49408)


def build_frame_captioner(
        frame_encoder: str = "clip_vitb16",
        clip_path: Union[str, Path] = "checkpoints/clip/ViT-B-16.pt",
        max_t_len: int = MAX_T_LEN,
        device: Optional[torch.device] = None,
) -> FrameVideoCaptioner:
    """Build the frame captioner: encoder + pretrained-initialized caption head.

    frame_encoder: "clip_vitb16" (CLIP ViT-B/16, 512-d native, CoCap's own
    backbone) or "maxvit_s" (project MaxViT-S, 768 -> 512 projection).
    The caption head is initialized from the official CaptionHead.from_pretrained
    (CLIP token embeddings + tied decoder), so the decoder is not random.
    """
    if frame_encoder == "clip_vitb16":
        encoder = CLIPFrameEncoder(clip_path=clip_path)
    elif frame_encoder == "maxvit_s":
        encoder = MaxViTFrameEncoder(feature_dim=512, pretrained=True)
    else:
        raise ValueError(f"unknown frame_encoder: {frame_encoder}")

    # cls() is FrameCaptionHead here - classmethod constructs cls(...)
    head = FrameCaptionHead.from_pretrained(
        pretrained_clip_name_or_path=str(clip_path),
        max_v_len=8 * 2,   # placeholder; overwritten per forward with N
        max_t_len=max_t_len,
    )
    model = FrameVideoCaptioner(encoder, head)
    if device is not None:
        model = model.to(device)
    return model


def tokenize_captions(captions: List[str], max_t_len: int = MAX_T_LEN):
    """Mirror the official dataset text pipeline (dataset_msrvtt.py)."""
    B = len(captions)
    input_ids = torch.zeros(B, max_t_len, dtype=torch.long)
    input_mask = torch.zeros(B, max_t_len, dtype=torch.long)
    for i, sentence in enumerate(captions):
        input_ids[i] = clip.tokenize(sentence, context_length=max_t_len, truncate=True)[0]
        input_mask[i, :len(clip._tokenizer.encode(sentence)) + 2] = 1
    return input_ids, input_mask


def decode_ids(tokens: List[int]) -> str:
    """Same detokenization as official lm_cocap.convert_ids_to_sentence
    (reimplemented here to avoid importing pytorch_lightning)."""
    from cocap.modules.clip.clip import _tokenizer
    text = _tokenizer.decode(tokens)
    text_list = text.split(" ")
    new = []
    for i in range(len(text_list)):
        if i == 0:
            new.append(text_list[i].split(">")[-1])
        elif "<|endoftext|>" in text_list[i]:
            break
        else:
            new.append(text_list[i])
    return " ".join(new)


@torch.no_grad()
def generate_caption(
        model: FrameVideoCaptioner,
        frames: Optional[Tensor] = None,
        frame_mask: Optional[Tensor] = None,
        visual_features: Optional[Tensor] = None,
        max_t_len: int = MAX_T_LEN,
) -> List[str]:
    """Greedy autoregressive decoding, mirroring official CoCapLM.validation_step:
    visual features computed ONCE, then one caption-head forward per token.
    """
    assert frames is not None or visual_features is not None
    device = frames.device if frames is not None else visual_features.device
    B = frames.shape[0] if frames is not None else visual_features.shape[0]

    if visual_features is None:
        B0, N, C, H, W = frames.shape
        visual_features = model.frame_encoder(frames.reshape(B0 * N, C, H, W)).reshape(B0, N, -1)

    input_ids = torch.zeros(B, max_t_len, dtype=torch.long, device=device)
    input_mask = torch.zeros(B, max_t_len, dtype=torch.long, device=device)
    next_symbols = torch.full((B,), model.caption_head.cap_config.BOS_id, dtype=torch.long)

    for dec_idx in range(max_t_len):
        input_ids[:, dec_idx] = next_symbols
        input_mask[:, dec_idx] = 1
        prediction_scores = model(
            visual_features=visual_features, frame_mask=frame_mask,
            input_ids=input_ids, input_mask=input_mask,
        )
        next_symbols = prediction_scores[:, dec_idx].max(1)[1].cpu()
        # early stop: tokens after the first EOS are discarded by the official
        # detokenizer anyway (convert_ids_to_sentence breaks at <|endoftext|>),
        # so ending the loop once EVERY sample has emitted EOS yields
        # byte-identical captions (~6x fewer decoder steps on short captions)
        if (next_symbols == model.caption_head.cap_config.EOS_id).all():
            break

    return [decode_ids(input_ids[i].tolist()) for i in range(B)]
