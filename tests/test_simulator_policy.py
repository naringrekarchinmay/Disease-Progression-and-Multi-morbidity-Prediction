"""Tests for the adaptive policy engine (research/simulator_policy.py, ISEM 735 W7-3).

The engine takes any scoring function, so these tests drive it with simple rules; the XGBoost risk
models are only built in the module's main (see tests/conftest.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import simulator as sim
from research import simulator_policy as pol
from research import simulator_sweep as sweep
from tests.test_simulator import params, source  # noqa: F401  (fixtures)


@pytest.fixture()
def cohort(source, params):  # noqa: F811
    return sweep.cohort_truth(source, params, n_patients=60, seed=5)


def always(features: pd.DataFrame) -> np.ndarray:
    return np.ones(len(features))


def test_measuring_every_quarter_matches_the_fixed_schedule_engine(cohort, source, params):  # noqa: F811
    run = pol.run_policy(cohort, source, params, beta=1.0, seed=5, score_fn=always, threshold=0.5)
    months = pd.DataFrame({"patient_id": np.repeat(cohort.ages.index, len(pol.DECISION_MONTHS)),
                           "month": np.tile(pol.DECISION_MONTHS, len(cohort.ages))})
    expected = sweep.run_schedule(cohort, sweep.visits_from_months(months), params, beta=1.0, seed=5)
    a = run.detected.set_index(["patient_id", "condition"])[["diagnosis_month", "route"]].sort_index()
    b = expected.set_index(["patient_id", "condition"])[["diagnosis_month", "route"]].sort_index()
    pd.testing.assert_frame_equal(a, b)
    assert len(run.visits) == len(cohort.ages) * len(pol.DECISION_MONTHS)


def test_in_loop_diagnoses_agree_with_final_detection(cohort, source, params):  # noqa: F811
    run = pol.run_policy(cohort, source, params, beta=1.0, seed=5, score_fn=always, threshold=0.5)
    lab = run.detected[run.detected["route"] == "lab"].set_index(["patient_id", "condition"])["diagnosis_month"]
    assert len(lab) > 0
    pd.testing.assert_series_equal(run.lab_dx_month.loc[lab.index].sort_index(), lab.sort_index(),
                                   check_names=False, check_dtype=False)


def test_never_measuring_leaves_only_clinical_diagnoses(cohort, source, params):  # noqa: F811
    run = pol.run_policy(cohort, source, params, beta=1.0, seed=5, score_fn=always, threshold=2.0)
    assert len(run.visits) == 0
    assert set(run.detected["route"]) <= {"clinical", "none"}


def test_features_use_only_results_from_before_the_decision(cohort, source, params):  # noqa: F811
    run = pol.run_policy(cohort, source, params, beta=1.0, seed=5, score_fn=always, threshold=0.5,
                         record=True)
    snaps = run.snapshots
    first = snaps[snaps["month"] == 0]
    assert first["last_glucose_fasting"].isna().all() and first["n_labs"].eq(0).all()
    later = snaps[snaps["month"] == 6].set_index("patient_id")
    labs = run.labs[(run.labs["test_name"] == "glucose_fasting") & (run.labs["month"] < 6)]
    last = labs.sort_values("month").groupby("patient_id")["value"].last()
    assert later.loc[last.index, "last_glucose_fasting"].tolist() == pytest.approx(last.tolist())
    assert (later["months_since_lab"] == 3).all()           # last visit was month 3
    assert (later["n_labs"] == 2).all()


def test_label_is_an_undiagnosed_onset_now_or_within_a_year(cohort, source, params):  # noqa: F811
    run = pol.run_policy(cohort, source, params, beta=1.0, seed=5, score_fn=always, threshold=2.0,
                         record=True)
    snaps = run.snapshots.set_index(["patient_id", "month"])
    truth = cohort.truth
    for (pid, d), row in snaps.sample(40, random_state=0).iterrows():
        t = truth[truth["patient_id"] == pid]
        known = t["clinical_month"] < d
        expected = bool(((t["onset_month"] <= d + 12) & ~known).any())
        assert bool(row["label"]) == expected


def test_random_quarterly_policy_hits_its_rate(cohort, source, params):  # noqa: F811
    score = pol.random_score(seed=3)
    run = pol.run_policy(cohort, source, params, beta=0.0, seed=5, score_fn=score, threshold=0.75)
    rate = len(run.visits) / (len(cohort.ages) * len(pol.DECISION_MONTHS))
    assert 0.18 < rate < 0.32
