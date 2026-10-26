"""Tests for the simulator smoke run (research/simulator_smoke.py, ISEM 735 W6-3).

The run itself fits XGBoost, so it is never called here (see tests/conftest.py); these tests
cover the pure-pandas pieces: which columns count as lab features and how the checks are judged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import simulator_smoke as smoke


def test_lab_columns_are_the_lab_prefixed_features():
    columns = ["patient_id", "age", "lab_test_count", "lab_mean_eGFR", "baseline_dx_asthma", "labelled"]
    assert smoke.lab_columns(columns) == ["lab_test_count", "lab_mean_eGFR"]


def _table(beta0: dict, beta1: dict) -> pd.DataFrame:
    rows = []
    for beta, aurocs in [(0.0, beta0), (1.0, beta1)]:
        rows += [{"beta": beta, "kind": kind, "name": name, "roc_auc": v}
                 for (kind, name), v in aurocs.items()]
    return pd.DataFrame(rows)


AUROCS_OK_0 = {("model", "all features"): 0.80, ("model", "no lab features"): 0.799,
               ("schedule", "keep all"): 0.80, ("schedule", "keep none"): 0.798, ("schedule", "12-month"): 0.799}
AUROCS_OK_1 = {("model", "all features"): 0.86, ("model", "no lab features"): 0.80,
               ("schedule", "keep all"): 0.86, ("schedule", "keep none"): 0.81, ("schedule", "12-month"): 0.84}


def test_checks_pass_when_labs_are_flat_at_beta_zero_and_informative_at_beta_one():
    checks = smoke.smoke_checks(_table(AUROCS_OK_0, AUROCS_OK_1)).set_index("check")
    assert checks["passed"].all(), checks


def test_checks_fail_when_labs_matter_at_beta_zero():
    leaky = AUROCS_OK_0 | {("model", "all features"): 0.83, ("schedule", "keep all"): 0.83}
    checks = smoke.smoke_checks(_table(leaky, AUROCS_OK_1)).set_index("check")
    assert not checks.loc["beta 0: lab features add nothing", "passed"]
    assert not checks.loc["beta 0: masking schedules tie", "passed"]
    assert checks.loc["beta 1: lab features add signal", "passed"]


def test_checks_fail_when_labs_carry_nothing_at_beta_one():
    flat = AUROCS_OK_1 | {("model", "all features"): 0.802, ("schedule", "keep all"): 0.802,
                          ("schedule", "keep none"): 0.80}
    checks = smoke.smoke_checks(_table(AUROCS_OK_0, flat)).set_index("check")
    assert not checks.loc["beta 1: lab features add signal", "passed"]
    assert not checks.loc["beta 1: keeping labs beats dropping them", "passed"]
