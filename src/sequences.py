"""Per-patient event sequences for the ISEM 735 sequence models.

Each patient becomes one time-ordered stream of events built from the five raw
tables, using only events on or before the cutoff date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import TARGET_COLUMNS, build_patient_features

EVENT_COLUMNS = ["patient_id", "date", "event_type", "token", "days_since_prev", "age_at_event"]
# Same-day events are ordered by type, then token, so the stream is deterministic.
EVENT_TYPE_ORDER = {"diagnosis": 0, "lab": 1, "medication": 2, "admission": 3}


def _diagnosis_events(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """One event per ICD-10 code per visit, primary and secondary codes alike."""
    codes = diagnoses["primary_icd10"] + "|" + diagnoses["secondary_icd10s"]
    exploded = diagnoses[["patient_id", "visit_date"]].assign(icd10=codes.str.split("|")).explode("icd10")
    exploded["icd10"] = exploded["icd10"].str.strip()
    exploded = exploded[exploded["icd10"].str.len() > 0].drop_duplicates()
    return pd.DataFrame(
        {
            "patient_id": exploded["patient_id"],
            "date": exploded["visit_date"],
            "event_type": "diagnosis",
            "token": "DX:" + exploded["icd10"],
        }
    )


def _lab_events(labs: pd.DataFrame) -> pd.DataFrame:
    """One event per lab test, with the value binned against the test's reference range."""
    level = np.select(
        [labs["value"] < labs["reference_low"], labs["value"] > labs["reference_high"]],
        ["low", "high"],
        default="normal",
    )
    return pd.DataFrame(
        {
            "patient_id": labs["patient_id"],
            "date": labs["test_date"],
            "event_type": "lab",
            "token": "LAB:" + labs["test_name"] + ":" + level,
        }
    )


def _medication_events(medications: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": medications["patient_id"],
            "date": medications["start_date"],
            "event_type": "medication",
            "token": "MED:" + medications["indication"],
        }
    )


def _admission_events(outcomes: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": outcomes["patient_id"],
            "date": outcomes["admission_date"],
            "event_type": "admission",
            "token": "ADM",
        }
    )


def build_sequences(
    data: dict[str, pd.DataFrame],
    cutoff_date: str = "2021-12-31",
    age_reference_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Return one row per input event, for events on or before the cutoff.

    `data` is the dictionary returned by `clean_all(load_raw_data())`. The patients
    table has one age per patient with no stated reference date; it is taken as the
    age on `age_reference_date`, the end of the data.
    """
    cutoff = pd.Timestamp(cutoff_date)
    events = pd.concat(
        [
            _diagnosis_events(data["diagnoses"]),
            _lab_events(data["lab_results"]),
            _medication_events(data["medications"]),
            _admission_events(data["outcomes"]),
        ],
        ignore_index=True,
    )
    events = events[events["date"] <= cutoff].copy()
    events["_type_rank"] = events["event_type"].map(EVENT_TYPE_ORDER)
    events = events.sort_values(["patient_id", "date", "_type_rank", "token"], kind="stable")

    events["days_since_prev"] = events.groupby("patient_id")["date"].diff().dt.days.fillna(0).astype(int)

    ages = data["patients"].set_index("patient_id")["age"]
    years_before_reference = (pd.Timestamp(age_reference_date) - events["date"]).dt.days / 365.25
    events["age_at_event"] = events["patient_id"].map(ages) - years_before_reference

    return events.reset_index(drop=True)[EVENT_COLUMNS]


def build_targets(
    data: dict[str, pd.DataFrame],
    cutoff_date: str = "2021-12-31",
    horizon_end: str = "2024-12-31",
) -> pd.DataFrame:
    """Return one row per patient with the same targets as the baseline feature table.

    Targets come from `build_patient_features`, so the sequence models and the
    XGBoost baseline predict exactly the same labels.
    """
    features = build_patient_features(
        patients=data["patients"],
        diagnoses=data["diagnoses"],
        lab_results=data["lab_results"],
        medications=data["medications"],
        outcomes=data["outcomes"],
        cutoff_date=cutoff_date,
        horizon_end=horizon_end,
    )
    return features[["patient_id", *TARGET_COLUMNS]]
