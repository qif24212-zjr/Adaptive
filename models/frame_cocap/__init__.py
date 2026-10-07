
frame_cocap: RGB frame-level captioning path on top of the official CoCap
backbone (third_party/CoCap). Official code is never modified; the original
compressed-domain path remains fully intact.

    frames [B, N, 3, 224, 224]  (N variable)
        -> FrameEncoder -> [B, N, 512]
        -> FrameCaptionHead (CoCap CaptionHead, single visual stream)
        -> logits [B, 77, 49408]

from .frame_sampling import collate_frames, read_video_frames_cv2, uniform_sample_indices
from .frame_video_captioner import (
    CLIPFrameEncoder,
    FrameCaptionHead,
    FrameVideoCaptioner,
    MaxViTFrameEncoder,
    build_frame_captioner,
    decode_ids,
    generate_caption,
    tokenize_captions,
)

__all__ = [
    "FrameCaptionHead",
    "CLIPFrameEncoder",
    "MaxViTFrameEncoder",
    "FrameVideoCaptioner",
    "build_frame_captioner",
    "tokenize_captions",
    "decode_ids",
    "generate_caption",
    "uniform_sample_indices",
    "collate_frames",
    "read_video_frames_cv2",
]
