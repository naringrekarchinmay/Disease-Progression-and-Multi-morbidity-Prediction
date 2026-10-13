"""Tests for the leak-free research data (research/data.py).

Run with `pytest tests/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.data import align_medications_to_diagnoses


def test_medications_cannot_start_before_their_condition_is_diagnosed():
    diagnoses = pd.DataFrame({
        "patient_id": ["p1", "p1"],
        "visit_date": pd.to_datetime(["2020-06-01", "2019-01-01"]),
        "visit_type": ["outpatient", "outpatient"],
        "primary_diagnosis": ["type2_diabetes", "hypertension"],
        "primary_icd10": ["E11.9", "I10"],
        "secondary_diagnoses": ["", ""],
        "secondary_icd10s": ["", ""],
        "provider_specialty": ["internal_medicine", "internal_medicine"],
    })
    medications = pd.DataFrame({
        "patient_id": ["p1", "p1", "p1"],
        "medication": ["Metformin", "Lisinopril", "Albuterol"],
        "indication": ["type2_diabetes", "hypertension", "asthma"],
        "start_date": pd.to_datetime(["2018-03-01", "2019-05-01", "2020-01-01"]),
    })
    result = align_medications_to_diagnoses(medications, diagnoses)
    assert list(result.columns) == list(medications.columns)
    aligned = result.set_index("medication")
    # Started 2018, diagnosed 2020: the start moves to the diagnosis date.
    assert aligned.loc["Metformin", "start_date"] == pd.Timestamp("2020-06-01")
    # Started after its diagnosis: unchanged.
    assert aligned.loc["Lisinopril", "start_date"] == pd.Timestamp("2019-05-01")
    # Asthma is never diagnosed in the record: the medication is dropped.
    assert "Albuterol" not in aligned.index


ROOT = Path(__file__).resolve().parents[1]


def test_leak_free_data_has_no_medication_before_its_diagnosis_and_unchanged_targets():
    import pytest

    if not (ROOT / "data" / "medications.csv").exists():
        pytest.skip("Raw CSVs not present (they are gitignored).")
    from research.data import load_clean_data
    from src.data_cleaning import make_condition_events
    from src.features import TARGET_COLUMNS, build_patient_features

    data = load_clean_data()
    first = make_condition_events(data["diagnoses"]).groupby(["patient_id", "condition"])["visit_date"].min()
    meds = data["medications"].set_index(["patient_id", "indication"])
    assert (meds["start_date"] >= first.reindex(meds.index).to_numpy()).all()

    ids = set(data["patients"]["patient_id"].sample(2000, random_state=42))
    part = {k: v[v["patient_id"].isin(ids)] for k, v in data.items()}
    rebuilt = build_patient_features(part["patients"], part["diagnoses"], part["lab_results"],
                                     part["medications"], part["outcomes"]).set_index("patient_id")
    original = pd.read_csv(ROOT / "data" / "processed" / "modeling_dataset.csv",
                           usecols=["patient_id", *TARGET_COLUMNS]).set_index("patient_id")
    pd.testing.assert_frame_equal(rebuilt[TARGET_COLUMNS].sort_index(), original.loc[sorted(ids), TARGET_COLUMNS],
                                  check_dtype=False)
