"""Leakage tests for baseline calibration (research/calibration.py).

Calibration must be fitted on a validation split carved from training data, and
the test split must be the same one the baseline pipeline evaluates on.

Run with `pytest tests/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_pipeline as rp
from research.calibration import calibration_splits, fit_calibrated_xgboost


def make_fake_features(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A small feature table with both baseline targets and one real signal."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    return pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            "age": rng.integers(20, 90, size=n),
            "signal": signal,
            "sex": rng.choice(["F", "M"], size=n),
            "baseline_condition_count": rng.integers(0, 2, size=n),
            "future_hospitalization": (signal + rng.normal(size=n) > 1).astype(int),
            "multimorbidity_progression": (signal + rng.normal(size=n) > 0.5).astype(int),
            "future_icu_admission": 0,
            "future_30d_readmission": 0,
            "new_condition_count": 0,
            "future_condition_count": 0,
        }
    )


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return make_fake_features()


@pytest.mark.parametrize("target", rp.TARGETS)
def test_splits_are_disjoint_and_test_matches_pipeline(features, target):
    splits = calibration_splits(features, target)
    fit_idx, val_idx, test_idx = (set(splits[name][0].index) for name in ["fit", "validation", "test"])
    assert fit_idx.isdisjoint(val_idx)
    assert test_idx.isdisjoint(fit_idx | val_idx)

    _, X_test, _, _ = rp.split_for_target(features, target)
    assert test_idx == set(X_test.index)
    # Fit and validation together are exactly the pipeline's training split.
    assert len(fit_idx | val_idx | test_idx) == len(rp.cohort_for_target(features, target))


@pytest.mark.parametrize("target", rp.TARGETS)
def test_calibrated_model_ignores_the_test_split(features, target):
    splits = calibration_splits(features, target)
    X_test, y_test = splits["test"]
    poisoned = {**splits, "test": (X_test.assign(signal=-X_test["signal"]), 1 - y_test)}

    _, calibrated = fit_calibrated_xgboost(splits)
    _, calibrated_poisoned = fit_calibrated_xgboost(poisoned)
    np.testing.assert_array_equal(
        calibrated.predict_proba(X_test)[:, 1], calibrated_poisoned.predict_proba(X_test)[:, 1]
    )
