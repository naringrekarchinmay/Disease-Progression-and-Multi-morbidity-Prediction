"""Tests for the paired bootstrap and DeLong helpers (research/stats_ci.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.stats_ci import bootstrap_indices, delong, paired_difference


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    strong = y + rng.normal(0, 0.8, n)
    weak = y + rng.normal(0, 2.0, n)
    return y, strong, weak


def test_delong_auc_matches_sklearn_including_ties():
    y, strong, weak = _data()
    tied = np.round(weak, 1)
    result = delong(y, strong, tied)
    assert abs(result["auc_a"] - roc_auc_score(y, strong)) < 1e-12
    assert abs(result["auc_b"] - roc_auc_score(y, tied)) < 1e-12


def test_delong_detects_a_real_difference_and_not_a_null_one():
    y, strong, weak = _data()
    assert delong(y, strong, weak)["p_value"] < 1e-6
    same = delong(y, strong, strong)
    assert same["diff"] == 0 and same["p_value"] == 1.0


def test_delong_se_agrees_with_the_paired_bootstrap():
    y, strong, weak = _data(n=2000, seed=1)
    boot = paired_difference(y, strong, weak, bootstrap_indices(len(y), 500, seed=2))
    assert abs(delong(y, strong, weak)["se"] - boot["boot_se"]) / boot["boot_se"] < 0.2


def test_paired_difference_ci_covers_the_estimate_and_averages_lists():
    y, strong, weak = _data(n=1000)
    idx = bootstrap_indices(len(y), 200)
    single = paired_difference(y, strong, weak, idx)
    assert single["ci_low"] < single["estimate"] < single["ci_high"] and single["share_above_zero"] == 1.0
    averaged = paired_difference(y, strong, [weak, weak], idx)
    assert abs(averaged["estimate"] - single["estimate"]) < 1e-12
