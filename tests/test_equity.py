"""Tests for the equity audit (research/equity.py).

Run with `pytest tests/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import numpy as np
from sklearn.metrics import roc_auc_score

from research.equity import measured_share, performance_by_group, subgroups


def test_age_bands_have_the_documented_edges():
    patients = pd.DataFrame(
        {
            "patient_id": ["a", "b", "c", "d", "e"],
            "age": [49, 50, 64, 65, 80],
            "sex": ["F", "M", "F", "M", "F"],
            "insurance_type": ["commercial", "medicaid", "medicare", "uninsured", "tricare"],
        }
    )
    groups = subgroups(patients).set_index("patient_id")
    assert groups["age_band"].astype(str).tolist() == ["<50", "50-64", "50-64", "65-79", "80+"]
    assert groups["insurance"].tolist() == ["commercial", "medicaid", "medicare", "uninsured", "tricare"]
    assert list(groups.columns) == ["sex", "age_band", "insurance"]


def test_measured_share_ranks_by_uncertainty_within_the_budget():
    df = pd.DataFrame(
        {
            "group": ["A"] * 4 + ["B"] * 6,
            "uncertainty": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        }
    )
    table = measured_share(df, "group", "uncertainty", budgets=[0.4, 0.5]).set_index(["budget", "group"])
    assert table.loc[(0.4, "A"), "selection_rate"] == 1.0
    assert table.loc[(0.4, "A"), "ratio_to_budget"] == pytest.approx(2.5)
    assert table.loc[(0.4, "B"), "selection_rate"] == 0.0
    assert table.loc[(0.5, "B"), "selection_rate"] == pytest.approx(1 / 6)
    assert table.loc[(0.5, "A"), "patients"] == 4


def test_performance_by_group_matches_sklearn_with_reproducible_intervals():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"group": rng.choice(["A", "B"], n), "y": y,
                       "score": np.clip(0.3 * y + rng.normal(0.35, 0.2, n), 0.01, 0.99)})
    table = performance_by_group(df, "group", "y", "score", n_boot=100, seed=7).set_index("group")
    for group, part in df.groupby("group"):
        row = table.loc[group]
        assert row["auroc"] == pytest.approx(roc_auc_score(part["y"], part["score"]))
        assert row["auroc_low"] <= row["auroc"] <= row["auroc_high"]
        assert row["calibration_gap"] == pytest.approx(part["y"].mean() - part["score"].mean())
    again = performance_by_group(df, "group", "y", "score", n_boot=100, seed=7).set_index("group")
    pd.testing.assert_frame_equal(table, again)
