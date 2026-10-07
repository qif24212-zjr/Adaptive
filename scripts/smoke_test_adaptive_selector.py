# -*- coding: utf-8 -*-
"""
Smoke test for the minimal Adaptive Selector (paper core).

    features = torch.randn(4, 32, 512)  ->  AdaptiveSelector  ->  SELECT / STOP

Checks (policy is randomly initialized, so selection QUALITY is not judged):
  1. forward succeeds
  2. action mask correct: strictly increasing indices, no duplicates
  3. temporal order preserved
  4. no STOP at step 0
  5. no STOP before min_selected_frames
  6. every episode either STOPs or hits max_selected_frames
  7. different samples can end with different selected counts

Run:  python scripts/smoke_test_adaptive_selector.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "xmodaler"))  # models.selectors.__init__ needs it

import torch

from models.selectors.adaptive_selector import AdaptiveSelector

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

B, M, D = 4, 32, 512
MIN_SEL, MAX_SEL = 2, 16


def run_and_check(selector, features, tag=""):
    out = selector(features)
    indices, mask, counts, stop_step = (
        out["selected_indices"], out["selected_mask"], out["selected_count"], out["stop_step"])

    print(f"--- {tag} ---")
    for b in range(features.shape[0]):
        idxs = indices[b, mask[b]].tolist()
        print(f"Video {b}:")
        print(f"    selected_indices = {idxs}")
        print(f"    selected_count = {counts[b].item()}")
        print(f"    stop_step = {stop_step[b].item()}")

    # assertions (the whole point of the smoke test)
    for b in range(features.shape[0]):
        idxs = indices[b, mask[b]].tolist()
        c = counts[b].item()
        s = stop_step[b].item()
        # 1. budget respected
        assert MIN_SEL <= c <= MAX_SEL, f"video {b}: count {c} outside [{MIN_SEL},{MAX_SEL}]"
        # 2/3. temporal order: strictly increasing
        assert all(idxs[i] < idxs[i + 1] for i in range(len(idxs) - 1)), \
            f"video {b}: indices not strictly increasing: {idxs}"
        # 3. no duplicates
        assert len(set(idxs)) == len(idxs), f"video {b}: duplicate indices: {idxs}"
        # 4/5. no stop at step 0, never before min_selected_frames
        assert s >= MIN_SEL, f"video {b}: stop_step {s} < min_selected_frames {MIN_SEL}"
        # 6. episode properly ended (STOP chosen or budget force-stop)
        assert s == c and s <= MAX_SEL, f"video {b}: stop_step {s} != count {c}"
    return counts


def main():
    print(f"device: {DEVICE} | input: ({B}, {M}, {D})")

    selector = AdaptiveSelector(
        feature_dim=D, hidden_size=256,
        min_selected_frames=MIN_SEL, max_selected_frames=MAX_SEL,
        sample=True,
    ).to(DEVICE)

    features = torch.randn(B, M, D, device=DEVICE)
    print("[1] forward with stochastic policy (sample=True)")
    counts = run_and_check(selector, features, "stochastic policy")

    print("\n[2] different batch samples can produce different selected counts")
    big = torch.randn(8, M, D, device=DEVICE)
    counts_big = selector(big)["selected_count"]
    print(f"    counts over 8 samples: {counts_big.tolist()}")
    assert len(set(counts_big.tolist())) >= 2, "all samples selected the same count!"
    print("    OK: variable selected counts across samples")

    print("\n[3] forward with argmax policy (sample=False)")
    selector.sample = False
    run_and_check(selector, features, "argmax policy")

    print("\n[4] stress across random init (20 fresh policies; also guards the\n"
          "    early-last-frame deadlock where the action mask could be all-forbidden)")
    for seed in range(1, 21):
        torch.manual_seed(seed)
        s2 = AdaptiveSelector(
            feature_dim=D, hidden_size=256,
            min_selected_frames=MIN_SEL, max_selected_frames=MAX_SEL,
            sample=True,
        ).to(DEVICE)
        c = s2(torch.randn(B, M, D, device=DEVICE))["selected_count"]
        assert all(MIN_SEL <= x <= MAX_SEL for x in c.tolist())
        print(f"    seed {seed}: counts {c.tolist()}")

    print("\nALL ADAPTIVE SELECTOR SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
