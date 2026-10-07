"""Base frame selector contract.

A selector consumes candidate frame features and returns a (possibly
video-dependent-length) selected subset. The captioner never sees the
selector: it only consumes `selected_features` + `selection_mask` through the
standard xmodaler ATT_FEATS / ATT_MASKS protocol (zero-padding + attention
mask handles variable-length selected sets).
"""
import torch
from torch import nn

__all__ = ["BaseSelector"]

# selection info keys carried in batched_inputs (plain strings; upstream kfg
# constants are intentionally not modified)
SEL_PROBS = "SELECTION_PROBS"
SEL_INDICES = "SELECTION_INDICES"
SEL_NUM_SELECTED = "NUM_SELECTED"
SEL_NUM_CANDIDATES = "NUM_CANDIDATES"
SEL_STATS = "SELECTION_STATS"


class BaseSelector(nn.Module):
    def forward(self, feats, masks):
        """
        Args:
            feats: (B, T, D) candidate frame features (padded).
            masks: (B, T) candidate validity mask (1 = valid candidate).
        Returns dict:
            selected_features: (B, T, D) dropped positions zeroed, temporal
                               order preserved (PickNet scan order).
            selection_mask:    (B, T) 1 = selected (never 1 at invalid
                               candidates; frame 0 is forced selected).
            selection_probs:   (B, T) P(PICK) used for the decision.
            selection_indices: list[list[int]] per-sample selected indices.
            num_selected:      (B,)
            num_candidates:    (B,)
            stats: dict with avg/min/max/median num_selected and selection
                   ratio (detached, for logging).
        """
        raise NotImplementedError
