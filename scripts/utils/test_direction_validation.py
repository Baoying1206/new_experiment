"""CPU-only unit tests for direction_validation.py's four held-out metrics,
extracted 2026-09-05 from scripts/26_rebuild_refusal_direction_behavioral.py
so scripts/23_extract_reference_directions.py can reuse them for
harmfulness_direction's validation. Hand-computed synthetic tensors only.

Usage:
  python scripts/utils/test_direction_validation.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from direction_validation import (
    cohens_d_per_layer, auc_per_layer, split_half_reliability, bootstrap_cohens_d_ci,
)


def test_cohens_d_hand_computed():
    # 1 layer. pos class = [2,2,2], neg class = [0,0,0] -> means differ by 2,
    # both classes have zero variance -> pooled_std uses the +1e-8 floor.
    proj = torch.tensor([[2.0], [2.0], [2.0], [0.0], [0.0], [0.0]])
    mask = torch.tensor([True, True, True, False, False, False])
    d = cohens_d_per_layer(proj, mask)
    assert d.shape == (1,)
    assert d[0] > 1000, f"near-zero-variance classes should give a huge Cohen's d, got {d[0]}"


def test_cohens_d_no_separation():
    proj = torch.tensor([[1.0], [2.0], [3.0], [1.0], [2.0], [3.0]])
    mask = torch.tensor([True, True, True, False, False, False])
    d = cohens_d_per_layer(proj, mask)
    assert abs(d[0].item()) < 1e-6, f"identical distributions should give d=0, got {d[0]}"


def test_auc_perfect_separation():
    proj = torch.tensor([[10.0], [11.0], [12.0], [1.0], [2.0], [3.0]])
    mask = torch.tensor([True, True, True, False, False, False])
    auc = auc_per_layer(proj, mask)
    assert abs(auc[0].item() - 1.0) < 1e-6, f"perfectly separated classes should give AUC=1.0, got {auc[0]}"


def test_auc_no_separation_class_empty():
    proj = torch.zeros(4, 1)
    mask = torch.tensor([True, True, True, True])  # no negative class at all
    auc = auc_per_layer(proj, mask)
    assert torch.isnan(auc[0]), "AUC with an empty class should be NaN, not fabricated"


def test_split_half_reliability_basic():
    torch.manual_seed(0)
    acts = torch.randn(20, 3, 8)  # [n, n_layers, d_model]
    mask = torch.tensor([True] * 10 + [False] * 10)
    cos = split_half_reliability(acts, mask, seed=0)
    assert cos is not None
    assert cos.shape == (3,)
    assert (cos >= -1.0001).all() and (cos <= 1.0001).all()


def test_split_half_reliability_too_few_per_class():
    acts = torch.randn(3, 2, 4)
    mask = torch.tensor([True, False, False])  # only 1 positive -- can't split
    assert split_half_reliability(acts, mask, seed=0) is None


def test_bootstrap_ci_empty_class_returns_none():
    proj = torch.randn(5, 2)
    mask = torch.tensor([True, True, True, True, True])  # no negative class
    lo, hi = bootstrap_cohens_d_ci(proj, mask, n_boot=50, seed=0)
    assert lo is None and hi is None


def test_bootstrap_ci_valid_bounds():
    torch.manual_seed(1)
    proj = torch.cat([torch.randn(20, 2) + 3.0, torch.randn(20, 2)], dim=0)
    mask = torch.cat([torch.ones(20, dtype=torch.bool), torch.zeros(20, dtype=torch.bool)])
    lo, hi = bootstrap_cohens_d_ci(proj, mask, n_boot=200, seed=0)
    assert lo is not None and hi is not None
    assert (lo <= hi).all()
    # a real, large effect (mean shift of 3, unit variance) should show up as a
    # clearly positive CI (both bounds well above 0), not spanning 0
    assert (lo > 0).all(), f"expected a clearly positive CI for a large true effect, got lo={lo}"


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
