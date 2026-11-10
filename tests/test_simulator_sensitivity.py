"""Tests for the pre-registered sensitivity sweep (research/simulator_sensitivity.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import simulator as sim
from research import simulator_mdp as mdp
from research import simulator_sensitivity as sens


def _params():
    return sim.SimulatorParams(labs=pd.DataFrame(), incidence={}, med_rate={}, med_pool=pd.DataFrame())


def test_scaling_matches_the_preregistered_levels():
    base = _params()
    assert sens.scaled_params(base, 1.0, 1.0).window == sim.WINDOW_MONTHS
    half = sens.scaled_params(base, 0.5, 1.5)
    assert half.window == {"type2_diabetes": 18, "chronic_kidney_disease": 24, "heart_failure": 15,
                           "hyperlipidemia": 12}
    assert half.clinical_lag["type2_diabetes"] == (36, 108)
    assert half.clinical_lag["heart_failure"] == (9, 36)
    assert half.clinical_lag["hyperlipidemia"] is None  # rule left alone
    assert base.window == sim.WINDOW_MONTHS  # the base params are not mutated


def test_components_reproduce_the_mdp_value_at_any_lambda():
    ids = pd.Index(["a", "b", "c"])
    detected = pd.DataFrame({"patient_id": ["a", "a", "b"],
                             "condition": ["type2_diabetes", "hyperlipidemia", "chronic_kidney_disease"],
                             "onset_month": [10.0, 5.0, 40.0], "diagnosis_month": [30.0, np.nan, 43.0]})
    visits = pd.DataFrame({"patient_id": ["a", "c", "c"],
                           "test_date": pd.to_datetime(["2019-01-15", "2018-03-01", "2023-06-01"])})
    comp = sens.components(detected, visits, ids)
    for lam in (0.0, 50.0, 400.0):
        expected = mdp.patient_values(detected, visits, ids, lam).to_numpy()
        np.testing.assert_allclose(-comp["C"] - lam * comp["U"], expected, rtol=1e-5)
    # Without hyperlipidemia, patient a loses its undiagnosed lipid months; b and c are unchanged.
    assert comp["U_nolipid"][0] < comp["U"][0]
    np.testing.assert_allclose(comp["U_nolipid"][1:], comp["U"][1:])


def _frame(rng, learned_gain, n=4000):
    cols = {}
    for p in sens.FIXED + [f"FQI {lam:g}" for lam in (100, 200, 400)]:
        cols[f"{p}|C"] = rng.uniform(0, 100, n)
        cols[f"{p}|U"] = rng.uniform(0, 10, n)
        cols[f"{p}|U_nolipid"] = cols[f"{p}|U"] * 0.8
    cols["FQI 200|U"] = cols["quarterly guideline|U"] - learned_gain
    cols["FQI 200|C"] = cols["quarterly guideline|C"]
    cols["quarterly guideline|U"] -= 2  # make the guideline the best fixed policy
    cols["FQI 200|U"] -= 2
    return pd.DataFrame(cols)


@pytest.mark.parametrize("gain, wins", [(1.0, True), (0.0, False), (-1.0, False)])
def test_primary_outcome_applies_the_preregistered_rule(gain, wins):
    rng = np.random.default_rng(0)
    frames = [_frame(rng, gain) for _ in range(3)]
    out = sens.primary_outcome(frames, 200.0, lipid=True)
    assert out["best_fixed"] == "quarterly guideline"
    assert abs(out["learned_minus_best"] - 200.0 * gain) < 1e-6
    assert out["adaptive_wins"] is wins
    assert out["ci_low"] <= out["learned_minus_best"] <= out["ci_high"]
