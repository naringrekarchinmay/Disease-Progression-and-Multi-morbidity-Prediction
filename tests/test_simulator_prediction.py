"""Tests for prediction under each schedule (research/simulator_prediction.py, ISEM 735 W7-2).

The model fitting loads XGBoost and is never called here (see tests/conftest.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import simulator as sim
from research import simulator_prediction as pred
from research import simulator_sweep as sweep
from tests.test_simulator import params, source  # noqa: F401  (fixtures)


def test_true_labels_count_onsets_not_diagnoses(source, params):  # noqa: F811
    cohort = sweep.cohort_truth(source, params, n_patients=30, seed=2)
    labels = pred.true_labels(cohort, source).set_index("patient_id")
    truth = cohort.truth.set_index(["patient_id", "condition"])["onset_month"]
    for pid, row in labels.iterrows():
        src = cohort.mapping.set_index("patient_id").loc[pid, "source_patient_id"]
        linked_base = sum(truth.get((pid, c), np.nan) <= pred.CUTOFF_MONTH for c in sim.LAB_CONDITIONS)
        linked_end = sum(truth.get((pid, c), np.nan) <= sim.N_MONTHS - 1 for c in sim.LAB_CONDITIONS)
        asthma = 1 if src == "P1" else 0                  # P1's asthma is diagnosed 2019-02-01
        assert row["true_baseline_count"] == linked_base + asthma
        assert row["true_future_count"] == linked_end + asthma
        assert row["at_risk"] == (row["true_baseline_count"] < 2)
        assert row["label"] == int(row["at_risk"] and row["true_future_count"] >= 2)


def test_labels_do_not_depend_on_the_schedule(source, params):  # noqa: F811
    cohort = sweep.cohort_truth(source, params, n_patients=30, seed=2)
    a = pred.true_labels(cohort, source)
    cohort.observed = cohort.observed.iloc[:0]        # no measurements at all
    pd.testing.assert_frame_equal(a, pred.true_labels(cohort, source))


def test_chunked_features_equal_one_pass(source, params):  # noqa: F811
    cohort = sweep.cohort_truth(source, params, n_patients=40, seed=3)
    one = pred.schedule_features(source, params, cohort, cohort.observed, beta=1.0, seed=3, chunk=1000)
    many = pred.schedule_features(source, params, cohort, cohort.observed, beta=1.0, seed=3, chunk=7)
    pd.testing.assert_frame_equal(one[sorted(one.columns)], many[sorted(many.columns)], check_dtype=False)


def test_features_only_use_visits_up_to_the_cutoff(source, params):  # noqa: F811
    cohort = sweep.cohort_truth(source, params, n_patients=20, seed=3)
    late = pd.DataFrame({"patient_id": cohort.ages.index, "test_date": pd.Timestamp("2023-06-15")})
    features = pred.schedule_features(source, params, cohort, late, beta=1.0, seed=3)
    assert (features["lab_test_count"] == 0).all()


def test_split_is_identical_for_identical_labels():
    labels = pd.DataFrame({"patient_id": [f"S{i:07d}" for i in range(400)], "at_risk": True,
                           "label": np.tile([0, 1, 1, 0, 0], 80)})
    a = pred.split_ids(labels, seed=42)
    b = pred.split_ids(labels.sample(frac=1, random_state=0), seed=42)
    for part in ["fit", "validation", "test"]:
        assert sorted(a[part]) == sorted(b[part])
    assert len(a["test"]) == 100
