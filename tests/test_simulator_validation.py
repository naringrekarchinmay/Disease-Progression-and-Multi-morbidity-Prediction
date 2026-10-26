"""Tests for simulator validation (research/simulator_validation.py, ISEM 735 W6-2).

The first group checks the metric code on hand-built tables. The second group simulates
20,000 patients at beta = 0 and checks each validation target against the Kaggle data within
its stated tolerance; it needs data/*.csv (gitignored) and takes about a minute.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import simulator as sim
from research import simulator_validation as val

DATA = Path(__file__).resolve().parents[1] / "data" / "patients.csv"


def _tables() -> dict[str, pd.DataFrame]:
    patients = pd.DataFrame({"patient_id": ["a", "b", "c"], "age": [40, 70, 85], "charlson_index": [0, 2, 4]})
    labs = pd.DataFrame({"patient_id": ["a", "a", "a", "b"],
                         "test_date": pd.to_datetime(["2019-01-01", "2019-01-01", "2020-01-01", "2021-06-01"]),
                         "test_name": ["WBC", "LDL", "WBC", "WBC"], "is_abnormal": [0, 1, 1, 0]})
    diagnoses = pd.DataFrame({
        "patient_id": ["a", "a", "b", "c"],
        "visit_date": pd.to_datetime(["2019-05-01", "2023-02-01", "2020-01-01", "2018-03-01"]),
        "visit_type": "outpatient",
        "primary_diagnosis": ["asthma", "type2_diabetes", "hyperlipidemia", "type2_diabetes"],
        "primary_icd10": "", "secondary_diagnoses": ["", "", "", "heart_failure"], "secondary_icd10s": "",
        "provider_specialty": "internal_medicine",
    })
    return {"patients": patients, "lab_results": labs, "diagnoses": diagnoses}


def test_lab_visit_metrics_count_patients_without_labs():
    m = val.lab_visit_metrics(_tables())
    assert m["lab_visits_per_patient"] == pytest.approx(1.0)       # 2 + 1 + 0 visits over 3 patients
    assert m["lab_visits_per_patient_median"] == 1.0
    assert m["median_gap_days"] == 365


def test_progression_uses_the_at_risk_cohort():
    # a: 1 condition by the cutoff, 2 by 2024 -> progresses. b: 1 and 1 -> no. c: 2 at baseline -> excluded.
    assert val.progression_rate(_tables()) == pytest.approx(0.5)


def test_condition_metrics_prevalence_and_incident_share():
    m = val.condition_metrics(_tables()).set_index(["metric", "group"])["value"]
    assert m[("prevalence", "type2_diabetes")] == pytest.approx(2 / 3)
    assert m[("incident_share", "type2_diabetes")] == pytest.approx(0.5)  # a after the cutoff, c before
    assert m[("prevalence_by_age", "type2_diabetes 80+")] == pytest.approx(1.0)
    assert m[("prevalence", "chronic_kidney_disease")] == 0.0


def test_validation_table_scores_within_tolerance_and_skips_signal_abnormal_rates():
    kaggle = pd.DataFrame({"metric": ["median_gap_days", "abnormal_rate", "abnormal_rate"],
                           "group": ["", "WBC", "eGFR"], "value": [431.0, 0.003, 0.9]})
    simulated = kaggle.assign(value=[450.0, 0.004, 0.0])
    table = val.validation_table(kaggle, simulated).set_index(["metric", "group"])
    assert table.loc[("median_gap_days", ""), "passed"] == 0.0       # 19 days off, tolerance 15
    assert table.loc[("abnormal_rate", "WBC"), "passed"] == 1.0
    assert not table.loc[("abnormal_rate", "eGFR"), "scored"]
    assert np.isnan(table.loc[("abnormal_rate", "eGFR"), "passed"])


# --- validation against the Kaggle data, beta = 0 ---------------------------------------------

KNOWN_MISSES = {
    # Kaggle's never-diabetic glucose averages 131 mg/dL (45% at or above 126), so at beta = 0 a
    # test after onset still triggers a correct diagnosis; post-2024 onsets get pulled into the window.
    ("prevalence", "type2_diabetes"): "glucose false-trigger effect, research/simulator_validation.md",
}


@pytest.fixture(scope="module")
def validation() -> pd.DataFrame:
    if not DATA.exists():
        pytest.skip("Kaggle CSVs not present (they are gitignored).")
    source = sim.load_source()
    params = sim.fit_params(source)
    tables, _ = sim.simulate_cohort(source, params, n_patients=20000, beta=0.0, seed=42)
    return val.validation_table(val.metric_frame(source), val.metric_frame(tables)).set_index(["metric", "group"])


TARGETS = ([(m, "") for m in ["lab_visits_per_patient", "lab_visits_per_patient_median", "median_gap_days",
                              "charlson_spearman", "progression_rate"]]
           + [(m, c) for m in ["prevalence", "incident_share"] for c in sim.LAB_CONDITIONS]
           + [("prevalence_by_age", f"{c} {b}") for c in sim.LAB_CONDITIONS for b in val.AGE_LABELS]
           + [("abnormal_rate", t) for t in ["WBC", "hemoglobin", "sodium", "potassium", "ALT", "AST", "HDL",
                                             "triglycerides", "troponin_I", "TSH"]])


@pytest.mark.parametrize("metric,group", TARGETS, ids=[f"{m}[{g}]" if g else m for m, g in TARGETS])
def test_beta_zero_matches_kaggle_within_tolerance(validation, metric, group):
    if (metric, group) in KNOWN_MISSES:
        pytest.xfail(KNOWN_MISSES[(metric, group)])
    row = validation.loc[(metric, group)]
    assert abs(row["value_sim"] - row["value_kaggle"]) <= row["tolerance"], row.to_dict()
