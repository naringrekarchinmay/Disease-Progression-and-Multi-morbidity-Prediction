"""Tests for the injected under-observation experiment (research/simulator_equity.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import simulator_mdp as mdp
from research import simulator_equity as eq
from research import simulator_sweep as sweep


def _cohort(n=4000):
    ids = pd.Index([f"S{i + 1:07d}" for i in range(n)])
    return sweep.Cohort(truth=pd.DataFrame(), ages=pd.Series(60.0, index=ids), observed=pd.DataFrame(),
                        mapping=pd.DataFrame())


def _features(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"insurance": rng.integers(0, 5, n), "known_type2_diabetes": 0,
                         "known_chronic_kidney_disease": 0, "age": 60.0, "n_labs": 1,
                         "months_since_lab": rng.integers(0, 60, n), "month": 24})


def test_rho_one_reproduces_the_behavior_policy_exactly():
    cohort, f = _cohort(), _features()
    a, pa = mdp.behavior_policy(cohort, 7)(f)
    b, pb = eq.under_observed_behavior(cohort, 7, 1.0)(f)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(pa, pb)


def test_rho_scales_only_the_subgroups_probability():
    cohort, f = _cohort(), _features()
    _, base = mdp.behavior_policy(cohort, 7)(f)
    _, half = eq.under_observed_behavior(cohort, 7, 0.5)(f)
    sub = f["insurance"].isin(eq.SUBGROUP_CODES).to_numpy()
    np.testing.assert_allclose(half[sub], 0.5 * base[sub])
    np.testing.assert_allclose(half[~sub], base[~sub])


def test_thinning_keeps_about_rho_of_subgroup_visits_and_all_others():
    n = 20000
    visits = pd.DataFrame({"patient_id": np.where(np.arange(n) % 2 == 0, "A", "B"),
                           "test_date": pd.Timestamp("2020-01-01")})
    out = eq.thin_visits(visits, {"A"}, 0.25, seed=1)
    assert (out["patient_id"] == "B").sum() == n // 2
    assert abs((out["patient_id"] == "A").sum() / (n // 2) - 0.25) < 0.02
    assert eq.thin_visits(visits, {"A"}, 1.0, seed=1) is visits
