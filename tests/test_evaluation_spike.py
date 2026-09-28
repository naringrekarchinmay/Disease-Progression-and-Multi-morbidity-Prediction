"""Tests for the Task 9 evaluation spike (research/evaluation_spike.py).

Run with `pytest tests/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.evaluation_spike import follows_schedule, hide_lab_visit


def lab_days(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame({"patient_id": [r[0] for r in rows], "date": pd.to_datetime([r[1] for r in rows])})


def test_follows_schedule_needs_every_gap_near_the_interval():
    days = lab_days(
        [
            ("quarterly", "2020-01-01"), ("quarterly", "2020-03-31"), ("quarterly", "2020-07-04"),  # 90, 95
            ("broken", "2020-01-01"), ("broken", "2020-03-31"), ("broken", "2020-10-17"),  # 90, 200
            ("one_gap", "2020-01-01"), ("one_gap", "2020-03-29"),  # 88, but only one gap
            ("yearly", "2019-01-01"), ("yearly", "2020-01-01"), ("yearly", "2021-01-05"),  # 365, 370
        ]
    )
    quarterly = follows_schedule(days, interval_days=91)
    yearly = follows_schedule(days, interval_days=365)
    assert quarterly[quarterly].index.tolist() == ["quarterly"]
    assert yearly[yearly].index.tolist() == ["yearly"]
    assert set(quarterly.index) == {"quarterly", "broken", "one_gap", "yearly"}


def make_events() -> pd.DataFrame:
    rows = [
        ("p1", "2019-05-01", "lab", "LAB:hba1c:high"),
        ("p1", "2020-02-01", "lab", "LAB:hba1c:normal"),
        ("p1", "2020-02-01", "lab", "LAB:WBC:normal"),
        ("p1", "2020-02-01", "diagnosis", "DX:I10"),
        ("p2", "2020-06-01", "medication", "MED:asthma"),
    ]
    return pd.DataFrame(rows, columns=["patient_id", "date", "event_type", "token"]).assign(
        date=lambda d: pd.to_datetime(d["date"])
    )


def test_hide_last_lab_visit_keeps_other_events_that_day():
    hidden = hide_lab_visit(make_events(), which="last")
    p1 = hidden[hidden["patient_id"] == "p1"]
    assert sorted(p1["token"]) == ["DX:I10", "LAB:hba1c:high"]
    # p2 has no labs, so nothing changes.
    assert hidden[hidden["patient_id"] == "p2"]["token"].tolist() == ["MED:asthma"]


def test_hide_random_lab_visit_removes_exactly_one_lab_day_reproducibly():
    events = make_events()
    first = hide_lab_visit(events, which="random", seed=1)
    lab_days_left = first[(first["patient_id"] == "p1") & (first["event_type"] == "lab")]["date"].nunique()
    assert lab_days_left == 1
    pd.testing.assert_frame_equal(first, hide_lab_visit(events, which="random", seed=1))
