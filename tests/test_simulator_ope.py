"""Tests for the off-policy estimators (research/simulator_ope.py, ISEM 735 W9-3).

The regressor is pluggable; these tests use scikit-learn trees on a toy logged dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import simulator_ope as ope
from tests.test_simulator_fqi import toy_logged

COLS = ["month", "sick", "found"]
G = 0.99


def tree():
    return DecisionTreeRegressor(max_depth=4, min_samples_leaf=5)


def logged_with_propensity(n=600, quarters=3, seed=0):
    df = toy_logged(n=n, quarters=quarters, seed=seed)
    df["p_measure"] = 0.5                    # toy actions are coin flips
    return df


def returns(df, reward, gamma=G):
    return (reward * gamma ** df["quarter"]).groupby(df["patient_id"]).sum()


def test_importance_weights_for_a_deterministic_target():
    df = logged_with_propensity(n=50)
    never = np.zeros(len(df))
    w = ope.cumulative_weights(df, never)
    # Weight is 2^(t+1) while the logged actions all match "never measure", and 0 after the first measurement.
    for pid, part in df.assign(w=w).groupby("patient_id"):
        matched = (part["action"] == 0).cumprod().to_numpy()
        assert part["w"].to_numpy() == pytest.approx(matched * 2.0 ** (np.arange(len(part)) + 1))


def test_on_policy_estimates_equal_the_sample_mean_return():
    df = logged_with_propensity()
    r = -df["undiagnosed_months"].to_numpy(dtype=float) - 2.0 * df["visits"].to_numpy()
    target = df["p_measure"].to_numpy()                      # evaluate the behavior policy itself
    w = ope.cumulative_weights(df, target)
    assert np.allclose(w, 1.0)
    mean_return = returns(df, r).mean()
    assert ope.wpdis(df, r, w, gamma=G) == pytest.approx(mean_return)


def test_wdr_reduces_to_wpdis_with_a_zero_model():
    df = logged_with_propensity()
    r = -df["undiagnosed_months"].to_numpy(dtype=float)
    target = np.zeros(len(df))
    w = ope.cumulative_weights(df, target)
    zeros = np.zeros(len(df))
    assert ope.wdr(df, r, w, zeros, zeros, gamma=G) == pytest.approx(ope.wpdis(df, r, w, gamma=G))


def test_fqe_recovers_values_when_reward_depends_only_on_the_action():
    df = logged_with_propensity()
    r = -10.0 * df["visits"].to_numpy(dtype=float)
    always = lambda states: np.ones(len(states))            # noqa: E731
    never = lambda states: np.zeros(len(states))             # noqa: E731
    for pi, expected in [(never, 0.0), (always, -10.0 * (1 + G + G**2))]:
        fit = ope.fqe(df, r, pi, COLS, tree, n_quarters=3, gamma=G)
        assert fit.value == pytest.approx(expected, abs=1e-6)


def test_fqe_and_wdr_track_the_truth_on_the_toy_problem():
    df = logged_with_propensity(n=3000)
    r = -df["undiagnosed_months"].to_numpy(dtype=float) - 2.0 * df["visits"].to_numpy()
    always = lambda states: np.ones(len(states))            # noqa: E731
    # Truth for "always measure": sick patients lose 3 months in quarter 0 only; everyone pays 2 a quarter.
    sick_share = df.loc[df["quarter"] == 0, "sick"].mean()
    truth = -sick_share * 3 - 2 * (1 + G + G**2)
    fit = ope.fqe(df, r, always, COLS, tree, n_quarters=3, gamma=G)
    w = ope.cumulative_weights(df, always(df))
    assert fit.value == pytest.approx(truth, abs=0.05)
    assert ope.wdr(df, r, w, fit.q_taken, fit.v_state, gamma=G) == pytest.approx(truth, abs=0.1)
    assert ope.wpdis(df, r, w, gamma=G) == pytest.approx(truth, abs=0.3)


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    df = logged_with_propensity(n=400)
    r = -df["undiagnosed_months"].to_numpy(dtype=float)
    w = ope.cumulative_weights(df, np.ones(len(df)))
    est = ope.wpdis(df, r, w, gamma=G)
    a = ope.bootstrap(lambda idx: ope.wpdis(df, r, w, gamma=G, patients=idx), df, n_boot=50, seed=1)
    b = ope.bootstrap(lambda idx: ope.wpdis(df, r, w, gamma=G, patients=idx), df, n_boot=50, seed=1)
    assert a == b and a[0] <= est <= a[1]


def test_effective_sample_size():
    assert ope.ess(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(4.0)
    assert ope.ess(np.array([4.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)
