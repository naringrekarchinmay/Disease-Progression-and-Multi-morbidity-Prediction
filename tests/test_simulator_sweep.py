"""Tests for the schedule sweep engine (research/simulator_sweep.py, ISEM 735 W7-1).

No XGBoost: the sweep only simulates measurements and detection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import simulator as sim
from research import simulator_sweep as sweep
from tests.test_simulator import params, source  # noqa: F401  (fixtures)


def test_common_random_numbers_depend_only_on_the_key():
    pnum = np.array([1, 2, 1, 1])
    month = np.array([10, 10, 10, 11])
    test = np.array([0, 0, 0, 0])
    z = sweep.crn_normal(pnum, month, test, seed=42)
    assert z[0] == z[2]                         # same patient, month, test -> same draw
    assert z[0] != z[1] and z[0] != z[3]
    assert sweep.crn_normal(pnum, month, test, seed=42).tolist() == z.tolist()
    big = sweep.crn_normal(np.arange(200000), np.zeros(200000, int), np.zeros(200000, int), seed=1)
    assert abs(big.mean()) < 0.01 and abs(big.std() - 1) < 0.01


def test_fixed_schedule_spaces_measurements_by_the_interval():
    ids = pd.Series(["S0000001", "S0000002"])
    months = sweep.fixed_schedule(ids, interval=12, seed=3)
    for pid, part in months.groupby("patient_id"):
        m = np.sort(part["month"].to_numpy())
        assert m[0] < 12 and np.all(np.diff(m) == 12) and m[-1] <= 83
    assert len(sweep.fixed_schedule(ids, interval=3, seed=3)) == 2 * 28


def test_random_schedule_matches_the_budget_without_repeats():
    ids = pd.Series([f"S{i:07d}" for i in range(1, 51)])
    months = sweep.random_schedule(ids, n_measurements=300, seed=5)
    assert len(months) == 300
    assert not months.duplicated(["patient_id", "month"]).any()
    assert months["month"].between(0, 83).all()


def test_guideline_intervals_follow_age_and_known_diagnoses():
    ages = pd.Series({"S1": 40, "S2": 25, "S3": 15, "S4": 50})
    # S4 is diagnosed with diabetes in month 30, so from then on it is measured yearly.
    dx_month = pd.DataFrame({"patient_id": ["S4"], "condition": ["type2_diabetes"], "diagnosis_month": [30.0]})
    months = sweep.guideline_schedule(ages, dx_month, phase=pd.Series({"S1": 0, "S2": 0, "S3": 0, "S4": 0}))
    by = months.groupby("patient_id")["month"].apply(list)
    assert by["S1"] == [0, 36, 72]               # ADA 2025: age 35 and over, every 3 years
    assert by["S2"] == [0, 60]                   # ACC/AHA 2013: risk factors every 4 to 6 years (5)
    assert "S3" not in by.index                  # under 20: no routine panel
    assert by["S4"] == [0, 36, 48, 60, 72]       # 36 is the first visit after diagnosis, then every 12


def test_measure_returns_trigger_tests_with_crn_values(params):  # noqa: F811
    truth = pd.DataFrame({"patient_id": ["S0000001"] * 4, "condition": list(sim.LAB_CONDITIONS),
                          "onset_month": [np.nan] * 4})
    visits = pd.DataFrame({"patient_id": ["S0000001", "S0000001"],
                           "test_date": pd.to_datetime(["2019-03-15", "2020-07-15"])})
    a = sweep.measure(visits, truth, params, beta=1.0, seed=9)
    b = sweep.measure(visits.iloc[[1]], truth, params, beta=1.0, seed=9)
    trigger_tests = [t for t in sim.TRIGGERS if t in params.labs.index]
    assert set(a["test_name"]) == set(trigger_tests)
    # The 2020 visit gets the same values whether or not the 2019 visit happened (common random numbers).
    later = a[a["test_date"] == pd.Timestamp("2020-07-15")].sort_values("test_name")["value"].to_numpy()
    assert later.tolist() == b.sort_values("test_name")["value"].tolist()


def test_restricted_delay_counts_onsets_in_the_window_and_censors_at_the_end():
    truth = pd.DataFrame({
        "patient_id": ["a", "b", "c", "d"], "condition": ["type2_diabetes"] * 4,
        "onset_month": [10.0, 70.0, -5.0, np.nan],
        "diagnosis_month": [22.0, np.nan, 3.0, np.nan],
        "route": ["lab", "none", "clinical", "none"],
    })
    out = sweep.delay_metrics(truth).set_index("condition").loc["type2_diabetes"]
    assert out["onsets_in_window"] == 2                       # a and b (c is prevalent, d never)
    assert out["restricted_mean_delay"] == pytest.approx((12 + 14) / 2)  # b censored at month 84
    assert out["diagnosed_share"] == pytest.approx(0.5)
    assert out["lab_detected_share"] == pytest.approx(0.5)


def test_denser_schedules_never_detect_later(source, params):  # noqa: F811
    cohort = sweep.cohort_truth(source, params, n_patients=200, seed=4)
    results = {}
    for interval in [3, 12]:
        months = sweep.fixed_schedule(cohort.truth["patient_id"].drop_duplicates(), interval, seed=4)
        results[interval] = sweep.run_schedule(cohort, months_to_visits(months), params, beta=1.0, seed=4)
    # With the same phase, every 12-month visit is also a 3-month visit, and noise is shared.
    a = results[3].set_index(["patient_id", "condition"])["diagnosis_month"].fillna(999)
    b = results[12].set_index(["patient_id", "condition"])["diagnosis_month"].fillna(999)
    assert (a <= b).all()


def months_to_visits(months: pd.DataFrame) -> pd.DataFrame:
    return sweep.visits_from_months(months)


def test_cohort_truth_matches_simulate_cohort_for_the_same_seed(source, params):  # noqa: F811
    cohort = sweep.cohort_truth(source, params, n_patients=50, seed=6)
    _, truth = sim.simulate_cohort(source, params, n_patients=50, beta=1.0, seed=6)
    a = cohort.truth.set_index(["patient_id", "condition"]).sort_index()
    b = truth.set_index(["patient_id", "condition"]).sort_index()
    pd.testing.assert_series_equal(a["onset_month"], b["onset_month"])
    pd.testing.assert_series_equal(a["clinical_date"], b["clinical_date"])
