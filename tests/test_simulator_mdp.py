"""Tests for the decision process used by week 9 (research/simulator_mdp.py, ISEM 735 W9-1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import simulator as sim
from research import simulator_mdp as mdp
from research import simulator_policy as pol
from research import simulator_sweep as sweep
from research.cost_model import SIMULATOR_MEASUREMENT_COST
from tests.test_simulator import params, source  # noqa: F401  (fixtures)

G = mdp.GAMMA


def test_discount_is_three_percent_a_year_per_quarter():
    assert G == pytest.approx(1.03 ** -0.25)
    assert G**4 == pytest.approx(1 / 1.03)


def test_patient_value_matches_a_hand_calculation():
    detected = pd.DataFrame({"patient_id": ["a", "a"], "condition": ["type2_diabetes", "heart_failure"],
                             "onset_month": [10.0, -5.0], "diagnosis_month": [22.0, np.nan]})
    visits = pd.DataFrame({"patient_id": ["a", "a"], "test_date": pd.to_datetime(["2018-04-15", "2019-04-15"])})
    lam = 50.0
    cost = SIMULATOR_MEASUREMENT_COST * (G**1 + G**5)                    # months 3 and 15
    diabetes = sum(G ** (m // 3) for m in range(10, 22))                 # undiagnosed months 10..21
    heart = sum(G ** (m // 3) for m in range(0, 84))                     # prevalent, never diagnosed
    value = mdp.patient_values(detected, visits, pd.Index(["a"]), lam).loc["a"]
    assert value == pytest.approx(-cost - lam * (diabetes + heart))


def test_quarter_components_add_up_to_the_value():
    detected = pd.DataFrame({"patient_id": ["a", "b"], "condition": ["chronic_kidney_disease"] * 2,
                             "onset_month": [30.0, np.nan], "diagnosis_month": [40.0, np.nan]})
    visits = pd.DataFrame({"patient_id": ["a", "b", "b"],
                           "test_date": pd.to_datetime(["2020-01-15", "2018-01-15", "2024-10-15"])})
    ids = pd.Index(["a", "b"])
    comp = mdp.quarter_components(detected, visits, ids)
    for lam in [25.0, 200.0]:
        rewards = -SIMULATOR_MEASUREMENT_COST * comp["visits"] - lam * comp["undiagnosed_months"]
        by_patient = (rewards * G ** comp["quarter"]).groupby(comp["patient_id"]).sum()
        pd.testing.assert_series_equal(by_patient.loc[ids], mdp.patient_values(detected, visits, ids, lam),
                                       check_names=False)
    assert comp.groupby("patient_id").size().tolist() == [28, 28]


def test_quarterly_guideline_rule():
    f = pd.DataFrame({"age": [40, 25, 15, 50, 50], "month": 12, "n_labs": [1, 1, 0, 1, 0],
                      "months_since_lab": [12, 12, np.nan, 12, np.nan],
                      "known_type2_diabetes": [0, 0, 0, 1, 0], "known_chronic_kidney_disease": 0})
    phase = np.array([0, 0, 0, 0, 15])
    act = mdp.guideline_actions(f, phase)
    # 40-year-old: 3-year interval, last lab 12 months ago -> wait; 25: 5-year -> wait; 15: never;
    # known diabetic: yearly -> measure; no lab yet but phase 15 > month 12 -> wait.
    assert act.tolist() == [False, False, False, True, False]
    f2 = f.assign(month=15)
    assert mdp.guideline_actions(f2, phase).tolist()[4]


def test_behavior_policy_propensities(source, params):  # noqa: F811
    cohort = sweep.cohort_truth(source, params, n_patients=200, seed=8)
    behavior = mdp.behavior_policy(cohort, seed=8)
    run = pol.run_policy(cohort, source, params, beta=1.0, seed=8, policy_fn=behavior, record=True)
    s = run.snapshots
    assert set(np.round(s["p_measure"], 6)) <= {0.85, 0.05}
    for p in [0.85, 0.05]:
        part = s[np.isclose(s["p_measure"], p)]
        if len(part) > 300:
            assert abs(part["action"].mean() - p) < 0.06


def test_logged_data_links_each_state_to_the_next(source, params):  # noqa: F811
    cohort = sweep.cohort_truth(source, params, n_patients=50, seed=9)
    logged = mdp.logged_dataset(cohort, source, params, beta=1.0, seed=9)
    assert len(logged) == 50 * 28
    a = logged[logged["quarter"] == 3].set_index("patient_id")
    b = logged[logged["quarter"] == 4].set_index("patient_id")
    assert (a["next_n_labs"] == b["n_labs"]).all()
    assert logged.loc[logged["quarter"] == 27, "terminal"].all()
    assert {"visits", "undiagnosed_months", "action", "p_measure"} <= set(logged.columns)
    assert (logged["visits"] == logged["action"]).all()


def test_true_value_of_a_schedule_is_the_mean_patient_value(source, params):  # noqa: F811
    cohort = sweep.cohort_truth(source, params, n_patients=40, seed=4)
    visits = sweep.visits_from_months(sweep.fixed_schedule(pd.Series(cohort.ages.index), 12, seed=4))
    detected = sweep.run_schedule(cohort, visits, params, beta=1.0, seed=4)
    values = mdp.true_values(detected, visits, cohort.ages.index, lams=[25.0, 50.0])
    assert values[50.0] == pytest.approx(mdp.patient_values(detected, visits, cohort.ages.index, 50.0).mean())
    assert values[25.0] > values[50.0]
