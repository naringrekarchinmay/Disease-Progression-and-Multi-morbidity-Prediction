"""Data loading and cleaning helpers for the synthetic EHR project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def find_data_file(filename: str, data_dir: Path | str = DATA_DIR) -> Path:
    """Find a raw CSV in data/raw first, then fall back to data/."""
    data_dir = Path(data_dir)
    candidates = [data_dir / "raw" / filename, data_dir / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename} in {data_dir}/raw or {data_dir}")


def load_raw_data(data_dir: Path | str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the five raw EHR CSV files into a dictionary of DataFrames."""
    files = {
        "patients": "patients.csv",
        "diagnoses": "diagnoses.csv",
        "lab_results": "lab_results.csv",
        "medications": "medications.csv",
        "outcomes": "outcomes.csv",
    }
    data = {}
    for name, filename in files.items():
        data[name] = pd.read_csv(find_data_file(filename, data_dir))
    return data


def clean_patients(patients: pd.DataFrame) -> pd.DataFrame:
    """Apply simple type fixes and clinical range checks to patient profiles."""
    df = patients.copy()
    df["patient_id"] = df["patient_id"].astype(str)

    numeric_cols = [
        "age",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "heart_rate",
        "temperature_f",
        "charlson_index",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep obvious impossible values as missing instead of silently modeling them.
    if "age" in df.columns:
        df.loc[~df["age"].between(0, 110), "age"] = pd.NA
    if "bmi" in df.columns:
        df.loc[~df["bmi"].between(10, 80), "bmi"] = pd.NA
    if "systolic_bp" in df.columns:
        df.loc[~df["systolic_bp"].between(60, 260), "systolic_bp"] = pd.NA
    if "diastolic_bp" in df.columns:
        df.loc[~df["diastolic_bp"].between(30, 160), "diastolic_bp"] = pd.NA

    return df


def clean_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Standardize diagnosis dates and text fields."""
    df = diagnoses.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")

    text_cols = [
        "visit_type",
        "primary_diagnosis",
        "primary_icd10",
        "secondary_diagnoses",
        "secondary_icd10s",
        "provider_specialty",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def clean_labs(labs: pd.DataFrame) -> pd.DataFrame:
    """Standardize lab dates and numeric values."""
    df = labs.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")
    for col in ["value", "reference_low", "reference_high", "is_abnormal", "delta_from_normal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_medications(medications: pd.DataFrame) -> pd.DataFrame:
    """Standardize medication dates and numeric fields."""
    df = medications.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    for col in ["dose", "duration_days", "is_generic", "adherence_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Standardize hospital outcome dates and numeric fields."""
    df = outcomes.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    for col in ["admission_date", "discharge_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    numeric_cols = [
        "length_of_stay_days",
        "icu_admission",
        "icu_days",
        "in_hospital_death",
        "readmitted_30d",
        "days_to_readmission",
        "primary_drg",
        "total_charges_usd",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_all(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Clean every raw table using project defaults."""
    return {
        "patients": clean_patients(data["patients"]),
        "diagnoses": clean_diagnoses(data["diagnoses"]),
        "lab_results": clean_labs(data["lab_results"]),
        "medications": clean_medications(data["medications"]),
        "outcomes": clean_outcomes(data["outcomes"]),
    }


def make_condition_events(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Convert primary and secondary diagnosis columns into one condition event table."""
    df = clean_diagnoses(diagnoses)

    primary = df[
        ["patient_id", "visit_date", "visit_type", "primary_diagnosis", "primary_icd10", "provider_specialty"]
    ].rename(columns={"primary_diagnosis": "condition", "primary_icd10": "icd10"})
    primary["diagnosis_position"] = "primary"

    secondary_rows = []
    secondary = df[df["secondary_diagnoses"].str.len() > 0].copy()
    for row in secondary.itertuples(index=False):
        conditions = str(row.secondary_diagnoses).split("|")
        codes = str(row.secondary_icd10s).split("|") if row.secondary_icd10s else []
        for i, condition in enumerate(conditions):
            condition = condition.strip()
            if not condition:
                continue
            secondary_rows.append(
                {
                    "patient_id": row.patient_id,
                    "visit_date": row.visit_date,
                    "visit_type": row.visit_type,
                    "condition": condition,
                    "icd10": codes[i].strip() if i < len(codes) else "",
                    "provider_specialty": row.provider_specialty,
                    "diagnosis_position": "secondary",
                }
            )

    secondary_events = pd.DataFrame(secondary_rows)
    events = pd.concat([primary, secondary_events], ignore_index=True)
    events = events.dropna(subset=["patient_id", "visit_date"])
    events = events[events["condition"].astype(str).str.len() > 0]
    return events.sort_values(["patient_id", "visit_date"]).reset_index(drop=True)


def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """Save a processed table under data/processed and return the path."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    return path
