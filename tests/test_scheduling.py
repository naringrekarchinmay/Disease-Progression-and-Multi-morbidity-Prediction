"""Tests for the masking-based scheduling experiments (research/scheduling.py).

Run with `pytest tests/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from research.scheduling import (
    features_as_of,
    fixed_interval_keep,
    lab_opportunities,
    random_keep,
    threshold_policy,
)
from src.features import TARGET_COLUMNS


def make_labs() -> pd.DataFrame:
    rows = [
        ("p1", "2018-03-01", "WBC"), ("p1", "2018-03-01", "sodium"),
        ("p1", "2018-09-01", "WBC"), ("p1", "2019-06-01", "WBC"),
        ("p1", "2022-02-01", "WBC"),  # after the cutoff: never a decision point
        ("p2", "2020-01-15", "HbA1c"),
    ]
    return pd.DataFrame(rows, columns=["patient_id", "test_date", "test_name"]).assign(
        test_date=lambda d: pd.to_datetime(d["test_date"]))


def test_opportunities_are_distinct_pre_cutoff_lab_days_in_order():
    opps = lab_opportunities(make_labs(), ["p1", "p2", "p3"])
    assert opps[["patient_id", "k"]].values.tolist() == [["p1", 0], ["p1", 1], ["p1", 2], ["p2", 0]]
    assert opps["date"].dt.strftime("%Y-%m-%d").tolist() == ["2018-03-01", "2018-09-01", "2019-06-01", "2020-01-15"]


def test_fixed_interval_keeps_visits_spaced_at_least_the_interval_less_30_days():
    opps = lab_opportunities(make_labs(), ["p1", "p2"])
    # p1 visits: 2018-03-01, 2018-09-01 (+184 d), 2019-06-01 (+273 d)
    assert fixed_interval_keep(opps, 182).tolist() == [True, True, True, True]
    assert fixed_interval_keep(opps, 365).tolist() == [True, False, True, True]


def test_random_keep_matches_the_budget_reproducibly():
    opps = lab_opportunities(make_labs(), ["p1", "p2"])
    kept = random_keep(opps, n_keep=2, seed=3)
    assert kept.sum() == 2 and len(kept) == len(opps)
    assert kept.tolist() == random_keep(opps, n_keep=2, seed=3).tolist()


def test_threshold_policy_decides_sequentially_on_risk_known_before_each_visit():
    opps = lab_opportunities(make_labs(), ["p1", "p2"])

    def risk_fn(opps, keep, k):
        # Risk drops by 0.1 for each visit already kept before visit k.
        at_k = opps[opps["k"] == k]
        kept_before = keep[(opps["k"] < k)].groupby(opps["patient_id"]).sum()
        return 0.5 - 0.1 * at_k["patient_id"].map(kept_before).fillna(0).set_axis(at_k["patient_id"])

    assert threshold_policy(opps, risk_fn, tau=0.15).tolist() == [True, True, False, True]
    assert threshold_policy(opps, risk_fn, tau=0.05).tolist() == [True, False, False, True]


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_sample():
    if not (ROOT / "data" / "patients.csv").exists() or not (ROOT / "data" / "processed" / "modeling_dataset.csv").exists():
        pytest.skip("Raw CSVs or modeling_dataset.csv not present (they are gitignored).")
    from src.data_cleaning import clean_all, load_raw_data

    raw = load_raw_data()
    ids = raw["patients"]["patient_id"].sample(300, random_state=42).tolist()
    data = clean_all({name: table[table["patient_id"].isin(set(ids))] for name, table in raw.items()})
    modeling = pd.read_csv(ROOT / "data" / "processed" / "modeling_dataset.csv")
    return ids, data, modeling[modeling["patient_id"].isin(set(ids))]


def test_features_at_the_cutoff_with_all_labs_reproduce_the_modeling_dataset(real_sample):
    ids, data, modeling = real_sample
    columns = [c for c in modeling.columns if c not in TARGET_COLUMNS]
    as_of = pd.Series(pd.Timestamp("2022-01-01"), index=ids)
    rebuilt = features_as_of(data, as_of, data["lab_results"], columns)
    expected = modeling.set_index("patient_id").loc[ids, columns[1:]].reset_index()
    pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True), expected, check_dtype=False)


def test_features_as_of_exclude_events_on_or_after_the_date(real_sample):
    ids, data, modeling = real_sample
    labs = data["lab_results"]
    pid = labs["patient_id"].value_counts().index[0]  # a patient with several lab rows
    first = labs.loc[labs["patient_id"] == pid, "test_date"].min()
    columns = [c for c in modeling.columns if c not in TARGET_COLUMNS]
    row = features_as_of(data, pd.Series(first, index=[pid]), labs, columns).iloc[0]
    assert row["lab_test_count"] == 0


def test_scheduling_module_never_loads_torch():
    """It loads XGBoost, which must never share a process with torch (see tests/conftest.py)."""
    import subprocess

    script = "import sys, research.scheduling; sys.exit('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, timeout=120)
    assert result.returncode == 0, "research.scheduling loaded torch"
