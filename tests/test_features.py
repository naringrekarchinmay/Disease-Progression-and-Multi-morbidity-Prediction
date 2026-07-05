"""Leakage tests for the feature engineering step.

These tests build a tiny fake EHR dataset and check that:
- the feature table contains the expected target columns
- the model matrix X never contains target or future information
- the date cutoff is respected (post-cutoff events only affect targets)

Run with `pytest tests/` or directly with `python tests/test_features.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import TARGET_COLUMNS, build_patient_features, get_modeling_matrices

CUTOFF = "2021-12-31"


def make_fake_raw_data() -> dict[str, pd.DataFrame]:
    """Three patients: one hospitalized only before the cutoff, one only after,
    and one who progresses to multimorbidity after the cutoff."""
    patients = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3"],
            "age": [70, 55, 40],
            "sex": ["F", "M", "F"],
            "bmi": [28.0, 31.5, 24.0],
            "dx_diabetes": [1, 0, 0],  # all-time flag, must be dropped from features
        }
    )
    diagnoses = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3", "p3"],
            "visit_date": ["2020-05-01", "2021-06-15", "2021-03-10", "2021-08-01", "2023-04-01"],
            "visit_type": ["outpatient"] * 5,
            "primary_diagnosis": ["Diabetes", "Hypertension", "Asthma", "Asthma", "Hypertension"],
            "primary_icd10": ["E11", "I10", "J45", "J45", "I10"],
            "secondary_diagnoses": ["", "", "", "", ""],
            "secondary_icd10s": ["", "", "", "", ""],
            "provider_specialty": ["internal medicine"] * 5,
        }
    )
    lab_results = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "test_date": ["2021-01-05", "2021-02-10"],
            "test_name": ["hba1c", "glucose"],
            "value": [7.5, 110.0],
            "is_abnormal": [1, 0],
            "delta_from_normal": [1.0, 0.0],
        }
    )
    medications = pd.DataFrame(
        {
            "patient_id": ["p1"],
            "start_date": ["2020-06-01"],
            "medication": ["metformin"],
            "indication": ["diabetes"],
            "adherence_pct": [85.0],
            "duration_days": [365.0],
            "is_generic": [1],
        }
    )
    outcomes = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "admission_date": ["2021-04-01", "2023-06-01"],  # p1 before cutoff, p2 after
            "length_of_stay_days": [3.0, 5.0],
            "icu_admission": [0, 1],
            "readmitted_30d": [0, 0],
        }
    )
    return {
        "patients": patients,
        "diagnoses": diagnoses,
        "lab_results": lab_results,
        "medications": medications,
        "outcomes": outcomes,
    }


def build_features() -> pd.DataFrame:
    data = make_fake_raw_data()
    return build_patient_features(
        patients=data["patients"],
        diagnoses=data["diagnoses"],
        lab_results=data["lab_results"],
        medications=data["medications"],
        outcomes=data["outcomes"],
        cutoff_date=CUTOFF,
    )


def test_feature_table_has_target_columns():
    features = build_features()
    for target in TARGET_COLUMNS:
        assert target in features.columns, f"Missing target column: {target}"


def test_model_matrix_excludes_targets_and_ids():
    features = build_features()
    X, y = get_modeling_matrices(features, target="future_hospitalization")
    for target in TARGET_COLUMNS:
        assert target not in X.columns, f"Leakage: target {target} found in X"
    assert "patient_id" not in X.columns
    assert len(X) == len(y) == len(features)


def test_no_future_columns_in_predictors():
    features = build_features()
    X, _ = get_modeling_matrices(features, target="multimorbidity_progression")
    future_cols = [col for col in X.columns if col.startswith("future_")]
    assert future_cols == [], f"Leakage: future columns found in X: {future_cols}"


def test_alltime_dx_flags_are_dropped():
    features = build_features()
    dx_cols = [col for col in features.columns if col.startswith("dx_")]
    assert dx_cols == [], f"All-time dx_* columns should be dropped: {dx_cols}"


def test_cutoff_respected_for_hospitalizations():
    features = build_features().set_index("patient_id")
    # p1 was hospitalized before the cutoff: prior count 1, no future target.
    assert features.loc["p1", "prior_hospitalization_count"] == 1
    assert features.loc["p1", "future_hospitalization"] == 0
    # p2 was hospitalized after the cutoff: no prior count, positive target.
    assert features.loc["p2", "prior_hospitalization_count"] == 0
    assert features.loc["p2", "future_hospitalization"] == 1
    assert features.loc["p2", "future_icu_admission"] == 1


def test_cutoff_respected_for_diagnoses():
    features = build_features().set_index("patient_id")
    # p3's second condition arrived after the cutoff: baseline count stays 1
    # and the post-cutoff diagnosis only shows up in the progression target.
    assert features.loc["p3", "baseline_condition_count"] == 1
    assert features.loc["p3", "multimorbidity_progression"] == 1
    # p3's post-cutoff hypertension must not appear as a baseline flag.
    assert features.loc["p3", "baseline_dx_hypertension"] == 0


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASSED {test.__name__}")
    print(f"\n{len(tests)} feature tests passed.")
