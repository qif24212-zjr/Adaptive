"""Selector-aware meta architecture for xmodaler (project-side, additive).

Selects frames between the dataset's candidate features and the captioner by
overriding `preprocess_batch`, which is the single chokepoint every xmodaler
trainer (XE / RL) and the eval/test path (greedy/beam) goes through
(audited call sites: engine/defaults.py:512,553, engine/rl_trainer.py:41, ...).
The captioner and decode strategies are untouched and only ever see
selected_features + selection_mask via the standard ATT_FEATS/ATT_MASKS
protocol.

Data flow:
    CapERADataset -> candidate features (B,10,768) + ATT_MASKS
      -> preprocess_batch (pad/cuda)
      -> [Selector] sequential Pick/Drop -> variable-length selected set
      -> ATT_FEATS (B,10,768 zero-padded) + ATT_MASKS (selection mask)
      -> visual_embed -> TransformerEncoder -> TransformerDecoder -> predictor
"""
import torch

from xmodaler.config import CfgNode as CN
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.modeling.meta_arch.build import META_ARCH_REGISTRY
from xmodaler.modeling.meta_arch.transformer_enc_dec import TransformerEncoderDecoder

from .selectors import build_selector
from .selectors.base_selector import (SEL_INDICES, SEL_NUM_CANDIDATES,
                                      SEL_NUM_SELECTED, SEL_PROBS, SEL_STATS)

__all__ = ["SelectorTransformerEncoderDecoder"]

# --------------------------------------------------------------------------
# Register the SELECTOR config schema at import time (project-side extension;
# upstream xmodaler/config/defaults.py is untouched). get_cfg() clones _C, so
# any process that imports this module before building the config accepts
# SELECTOR.* keys in the yaml.
# --------------------------------------------------------------------------
from xmodaler.config.defaults import _C  # noqa: E402

if not hasattr(_C, "SELECTOR"):
    _C.SELECTOR = CN()
    _C.SELECTOR.NAME = ""
    _C.SELECTOR.HIDDEN_SIZE = 256
    _C.SELECTOR.INIT_PICK_PROB = 0.75
    _C.SELECTOR.TARGET_RATIO = 0.5
    _C.SELECTOR.LAMBDA_BUDGET = 1.0


@META_ARCH_REGISTRY.register()
class SelectorTransformerEncoderDecoder(TransformerEncoderDecoder):
    @configurable
    def __init__(self, *, selector, **kwargs):
        super().__init__(**kwargs)
        self.selector = selector

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret["selector"] = build_selector(cfg)
        return ret

    @classmethod
    def add_config(cls, cfg, tmp_cfg):
        super().add_config(cfg, tmp_cfg)

    def preprocess_batch(self, batched_inputs):
        ret = super().preprocess_batch(batched_inputs)  # pad + masks + cuda

        feats = ret[kfg.ATT_FEATS]                     # (B, T, 768)
        masks = ret[kfg.ATT_MASKS]                     # (B, T)
        sel = self.selector(feats, masks)

        ret[kfg.ATT_FEATS] = sel["selected_features"]  # zero-padded selected
        ret[kfg.ATT_MASKS] = sel["selection_mask"]     # variable-length mask
        ret[SEL_PROBS] = sel["selection_probs"]
        ret[SEL_INDICES] = sel["selection_indices"]
        ret[SEL_NUM_SELECTED] = sel["num_selected"]
        ret[SEL_NUM_CANDIDATES] = sel["num_candidates"]
        ret[SEL_STATS] = sel["stats"]
        return ret
