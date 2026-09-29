"""Tests for the measurement cost model (research/cost_model.py).

Run with `pytest tests/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.cost_model import TEST_TO_CODE, lab_visit_cost

CMP_TESTS = ["glucose_fasting", "creatinine", "potassium", "sodium", "ALT", "AST"]
CBC_TESTS = ["WBC", "hemoglobin"]


def test_basic_visit_pays_draw_cbc_and_cmp_once():
    # 36415 draw $9.34 + 85025 CBC $7.77 + 80053 CMP $10.56 (CMS CLFS 2026 Q4)
    assert lab_visit_cost(CMP_TESTS + CBC_TESTS) == pytest.approx(27.67)


def test_panel_tests_are_billed_once_and_egfr_is_free():
    lipids = ["LDL", "HDL", "total_cholesterol", "triglycerides"]
    with_lipids = lab_visit_cost(CMP_TESTS + CBC_TESTS + lipids + ["eGFR"])
    assert with_lipids == pytest.approx(27.67 + 13.39)
    # Repeating a test on the same visit does not add a charge.
    assert lab_visit_cost(CMP_TESTS + CBC_TESTS + ["sodium", "WBC"]) == pytest.approx(27.67)


def test_unknown_test_name_is_an_error():
    with pytest.raises(KeyError):
        lab_visit_cost(["not_a_test"])


def test_every_lab_test_in_the_data_has_a_billing_code():
    labs = Path(__file__).resolve().parents[1] / "data" / "lab_results.csv"
    if not labs.exists():
        pytest.skip("lab_results.csv not present (it is gitignored).")
    import pandas as pd

    names = set(pd.read_csv(labs, usecols=["test_name"])["test_name"].unique())
    assert names <= set(TEST_TO_CODE), names - set(TEST_TO_CODE)
