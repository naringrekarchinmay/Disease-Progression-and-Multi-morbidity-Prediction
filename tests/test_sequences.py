"""Leakage tests for the event-sequence builder (src/sequences.py).

The fake-data tests check that no input event falls after the cutoff and that each
table becomes the right tokens. The real-data tests compare against
data/processed/modeling_dataset.csv on a patient sample and are skipped when the
gitignored CSVs are not present.

Run with `pytest tests/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_cleaning import clean_all, load_raw_data
from src.features import TARGET_COLUMNS
from src.sequences import (
    PAD_ID,
    UNK_ID,
    build_sequences,
    build_targets,
    build_vocabulary,
    save_sequences,
)

CUTOFF = pd.Timestamp("2021-12-31")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_DATASET = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"
SAMPLE_SIZE, SEED = 2000, 42


def make_fake_raw_data() -> dict[str, pd.DataFrame]:
    """Two patients with events on both sides of the cutoff in every table."""
    patients = pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "age": [70, 50],
            "sex": ["F", "M"],
            "charlson_index": [2, 0],
            "dx_hypertension": [1, 0],
        }
    )
    diagnoses = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2"],
            "visit_date": ["2020-01-10", "2022-03-01", "2021-12-31"],
            "visit_type": ["outpatient", "outpatient", "telehealth"],
            "primary_diagnosis": ["hypertension", "heart_failure", "asthma"],
            "primary_icd10": ["I10", "I50.9", "J45.909"],
            "secondary_diagnoses": ["type2_diabetes|obesity", "", ""],
            "secondary_icd10s": ["E11.9|E66.9", "", ""],
            "provider_specialty": ["internal_medicine", "cardiology", "pulmonology"],
        }
    )
    lab_results = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1", "p2"],
            "test_date": ["2020-01-10", "2020-01-10", "2023-05-05", "2021-06-01"],
            "test_name": ["hba1c", "potassium", "hba1c", "hba1c"],
            "value": [8.1, 3.1, 6.0, 5.0],
            "unit": ["%", "mmol/L", "%", "%"],
            "reference_low": [4.0, 3.5, 4.0, 4.0],
            "reference_high": [5.6, 5.0, 5.6, 5.6],
            "flag": ["high", "low", "high", "normal"],
            "is_abnormal": [1, 1, 1, 0],
            "delta_from_normal": [2.5, -0.4, 0.4, 0.0],
        }
    )
    medications = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2"],
            "medication": ["Metformin", "Furosemide", "Albuterol"],
            "dose": [500.0, 40.0, 90.0],
            "unit": ["mg", "mg", "mcg"],
            "frequency": ["daily", "daily", "prn"],
            "indication": ["type2_diabetes", "heart_failure", "asthma"],
            "start_date": ["2020-02-01", "2022-04-01", "2021-07-01"],
            "duration_days": [90, 30, 30],
            "is_generic": [1, 1, 1],
            "adherence_pct": [80.0, 70.0, 90.0],
        }
    )
    outcomes = pd.DataFrame(
        {
            "patient_id": ["p1", "p1"],
            "admission_date": ["2021-11-20", "2022-06-01"],
            "discharge_date": ["2021-11-25", "2022-06-04"],
            "length_of_stay_days": [5, 3],
            "icu_admission": [0, 1],
            "icu_days": [0, 2],
            "in_hospital_death": [0, 0],
            "discharge_disposition": ["home", "home"],
            "readmitted_30d": [0, 0],
            "days_to_readmission": [None, None],
            "primary_drg": [291, 291],
            "total_charges_usd": [9000.0, 12000.0],
        }
    )
    return clean_all(
        {
            "patients": patients,
            "diagnoses": diagnoses,
            "lab_results": lab_results,
            "medications": medications,
            "outcomes": outcomes,
        }
    )


@pytest.fixture(scope="module")
def fake_events() -> pd.DataFrame:
    return build_sequences(make_fake_raw_data())


def test_no_input_event_after_cutoff(fake_events):
    assert len(fake_events) > 0
    assert fake_events["date"].max() <= CUTOFF


def test_tables_become_expected_tokens(fake_events):
    tokens = {pid: sorted(g["token"]) for pid, g in fake_events.groupby("patient_id")}
    assert tokens["p1"] == sorted(
        ["DX:I10", "DX:E11.9", "DX:E66.9", "LAB:hba1c:high", "LAB:potassium:low", "MED:type2_diabetes", "ADM"]
    )
    # The diagnosis on the cutoff day itself is an input.
    assert tokens["p2"] == sorted(["LAB:hba1c:normal", "MED:asthma", "DX:J45.909"])


def test_time_gaps_and_age_at_event(fake_events):
    p1 = fake_events[fake_events["patient_id"] == "p1"]
    assert p1["date"].is_monotonic_increasing
    # Same-day events share a gap of 0; later gaps count days since the previous event of any type.
    assert p1["days_since_prev"].tolist() == [0, 0, 0, 0, 0, 22, 658]
    # Age in the data is taken as age on 2024-12-31, so the 2020-01-10 visit is ~4.97 years earlier.
    assert p1["age_at_event"].iloc[0] == pytest.approx(65.03, abs=0.01)
    assert p1["age_at_event"].iloc[-1] == pytest.approx(70 - 1137 / 365.25, abs=0.01)


def test_targets_one_row_per_patient_from_post_cutoff_events():
    targets = build_targets(make_fake_raw_data()).set_index("patient_id")
    assert sorted(targets.index) == ["p1", "p2"]
    assert list(targets.columns) == TARGET_COLUMNS
    # p1's 2022 ICU admission is a target; the 2021 admission is only an input.
    assert targets.loc["p1", "future_hospitalization"] == 1
    assert targets.loc["p1", "future_icu_admission"] == 1
    assert targets.loc["p2", "future_hospitalization"] == 0



def test_vocabulary_covers_every_token_with_reserved_ids(fake_events):
    vocab = build_vocabulary(fake_events)
    assert set(vocab["token"]) == set(fake_events["token"])
    assert vocab["token_id"].is_unique
    assert {PAD_ID, UNK_ID}.isdisjoint(vocab["token_id"])
    # Most frequent token first; counts match the events.
    assert vocab["count"].is_monotonic_decreasing
    assert vocab["count"].sum() == len(fake_events)


def test_saved_sequences_round_trip_with_token_ids(fake_events, tmp_path):
    data = make_fake_raw_data()
    paths = save_sequences(fake_events, build_vocabulary(fake_events), build_targets(data), tmp_path)
    saved = pd.read_parquet(paths["sequences"])
    vocab = pd.read_csv(paths["vocabulary"])
    targets = pd.read_parquet(paths["targets"])
    pd.testing.assert_frame_equal(saved.drop(columns="token_id"), fake_events)
    assert (saved["token"].map(vocab.set_index("token")["token_id"]) == saved["token_id"]).all()
    assert len(targets) == len(data["patients"])

@pytest.fixture(scope="module")
def real_sample():
    """Real data for a random patient sample, plus the matching modeling_dataset rows."""
    if not (PROJECT_ROOT / "data" / "patients.csv").exists() or not MODELING_DATASET.exists():
        pytest.skip("Raw CSVs or modeling_dataset.csv not present (they are gitignored).")
    raw = load_raw_data()
    ids = set(raw["patients"]["patient_id"].sample(SAMPLE_SIZE, random_state=SEED))
    data = clean_all({name: table[table["patient_id"].isin(ids)] for name, table in raw.items()})
    columns = ["patient_id", *TARGET_COLUMNS]
    modeling = pd.read_csv(MODELING_DATASET, usecols=columns)[columns]
    return ids, data, modeling[modeling["patient_id"].isin(ids)]


def test_real_targets_match_modeling_dataset(real_sample):
    _, data, modeling = real_sample
    targets = build_targets(data)
    pd.testing.assert_frame_equal(
        targets.sort_values("patient_id").reset_index(drop=True),
        modeling.sort_values("patient_id").reset_index(drop=True),
        check_dtype=False,
    )


def test_real_patient_counts_agree_and_no_leakage(real_sample):
    ids, data, modeling = real_sample
    events = build_sequences(data)
    targets = build_targets(data)
    assert len(targets) == len(modeling) == SAMPLE_SIZE
    assert set(events["patient_id"]) <= ids
    assert events["date"].max() <= CUTOFF
