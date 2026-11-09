"""Tests for the lab lead-time measure and the lab constraint (research/lead_time.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.lead_time import constrain_labs, lead_time_table

DX = pd.Timestamp("2023-01-01")
TESTS = {"creatinine": "chronic_kidney_disease"}


def _cohort(case_value, n=300, seed=0):
    """n cases diagnosed on DX with one lab every half year from 2018, plus n controls."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", "2022-12-01", freq="6MS")
    rows = []
    for i in range(n):
        for d in dates:
            years = (DX - d).days / 365.25
            rows.append(("case%d" % i, d, case_value(years) + rng.normal(0, 0.3)))
            rows.append(("ctrl%d" % i, d, 1.0 + rng.normal(0, 0.3)))
    labs = pd.DataFrame(rows, columns=["patient_id", "test_date", "value"]).assign(test_name="creatinine")
    first = pd.DataFrame({"patient_id": ["case%d" % i for i in range(n)],
                          "condition": "chronic_kidney_disease", "dx_date": DX})
    return labs, first


def test_a_rising_biomarker_separates_best_close_to_diagnosis():
    labs, first = _cohort(lambda years: 1.0 + max(0.0, 2 - years))  # rises over the last 2 years
    table = lead_time_table(labs, first, tests=TESTS).set_index("years_before")
    assert table.loc["0-1", "auroc"] > 0.9
    assert abs(table.loc["3-4", "auroc"] - 0.5) < 0.1
    assert table["spearman_value_vs_years"].iloc[0] < -0.3


def test_a_lifetime_flag_separates_equally_at_every_lead_time():
    labs, first = _cohort(lambda years: 3.0)  # the Kaggle pattern: abnormal whatever the timing
    table = lead_time_table(labs, first, tests=TESTS)
    assert table["auroc"].min() > 0.95
    assert table["auroc"].max() - table["auroc"].min() < 0.03


def test_only_pre_diagnosis_labs_of_incident_cases_are_scored():
    labs, first = _cohort(lambda years: 3.0, n=50)
    late = labs.assign(test_date=labs["test_date"] + pd.DateOffset(years=6))  # all after DX
    assert lead_time_table(late, first, tests=TESTS).empty
    prevalent = first.assign(dx_date=pd.Timestamp("2019-06-01"))  # before the incident window
    assert lead_time_table(labs, prevalent, tests=TESTS).empty


def test_constraint_replaces_only_case_labs_older_than_the_window():
    labs, first = _cohort(lambda years: 3.0, n=50)
    labs = labs.assign(reference_low=0.6, reference_high=1.3, flag="x", is_abnormal=-1, delta_from_normal=0.0)
    out = constrain_labs(labs, first, window_months={"chronic_kidney_disease": 24}, tests=TESTS, seed=1)
    is_case = labs["patient_id"].str.startswith("case")
    old = is_case & (labs["test_date"] < DX - pd.DateOffset(months=24))
    assert out.attrs["replaced"] == int(old.sum()) > 0
    # Controls and labs inside the window keep their values, dates and order.
    pd.testing.assert_frame_equal(out.loc[~old], labs.loc[~old])
    # Replaced values come from the control pool, so the artifact is gone there.
    assert set(out.loc[old, "value"]) <= set(labs.loc[~is_case, "value"])
    assert abs(out.loc[old, "value"].mean() - 1.0) < 0.1
    assert set(out.loc[old, "flag"]) <= {"low", "normal", "high"}
    assert (out.loc[old, "is_abnormal"] == (out.loc[old, "flag"] != "normal")).all()


def test_constraint_turns_a_lifetime_flag_into_a_window_signal():
    labs, first = _cohort(lambda years: 3.0, n=200)
    out = constrain_labs(labs, first, window_months={"chronic_kidney_disease": 24}, tests=TESTS)
    table = lead_time_table(out, first, tests=TESTS).set_index("years_before")
    assert table.loc["0-1", "auroc"] > 0.95
    assert abs(table.loc["3-4", "auroc"] - 0.5) < 0.1


def test_constraint_is_reproducible_by_seed():
    labs, first = _cohort(lambda years: 3.0, n=20)
    a = constrain_labs(labs, first, window_months={"chronic_kidney_disease": 24}, tests=TESTS, seed=5)
    b = constrain_labs(labs, first, window_months={"chronic_kidney_disease": 24}, tests=TESTS, seed=5)
    pd.testing.assert_frame_equal(a, b)
