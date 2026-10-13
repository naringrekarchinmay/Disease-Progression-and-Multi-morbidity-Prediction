"""Leak-free research data (found 2026-10-12).

The Kaggle generator writes future diagnoses into the medication history: for every
condition first diagnosed after the 2021-12-31 cutoff, its medication had already
started before the cutoff, a median of 742 days before the diagnosis. Any model that
reads medications therefore "predicts" future conditions from the drug list.

The research pipeline uses data where a medication cannot start before its condition
is diagnosed: earlier starts move to the diagnosis date, and medications for conditions
never diagnosed in the record are dropped. Labs are left as they are; labs showing a
disease before its diagnosis is the signal measurement scheduling depends on.

The original pipeline (src/, run_pipeline.py, models/*.joblib, notebooks) is unchanged
and still describes the original data. Build the leak-free artifacts with:

    python -m research.data
"""

from __future__ import annotations

import pandas as pd

import run_pipeline as rp
from src.data_cleaning import clean_all, load_raw_data, make_condition_events

PROCESSED = rp.PROJECT_ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED / "modeling_dataset_leakfree.csv"
SEQUENCE_SUFFIX = "_leakfree"
SEQUENCES_PATH = PROCESSED / f"sequences{SEQUENCE_SUFFIX}.parquet"


def align_medications_to_diagnoses(medications: pd.DataFrame, diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Move medication starts that precede their condition's first diagnosis to that date.

    Medications whose indication is never diagnosed for the patient are dropped.
    """
    first = (make_condition_events(diagnoses).groupby(["patient_id", "condition"])["visit_date"].min()
             .rename("first_diagnosis").reset_index().rename(columns={"condition": "indication"}))
    merged = medications.merge(first, on=["patient_id", "indication"], how="inner")
    merged["start_date"] = merged[["start_date", "first_diagnosis"]].max(axis=1)
    return merged[medications.columns].reset_index(drop=True)


def load_clean_data() -> dict[str, pd.DataFrame]:
    """The five cleaned tables with medications aligned to diagnoses."""
    data = clean_all(load_raw_data())
    data["medications"] = align_medications_to_diagnoses(data["medications"], data["diagnoses"])
    return data


def load_features(sample: int | None = None) -> pd.DataFrame:
    """Leak-free feature table, sampled like run_pipeline.load_feature_table."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"{FEATURES_PATH} not found. Run: python -m research.data")
    features = pd.read_csv(FEATURES_PATH)
    if sample is not None and sample < len(features):
        features = features.sample(n=sample, random_state=rp.RANDOM_STATE).reset_index(drop=True)
    return features


def main() -> None:
    import time

    from src.features import build_patient_features, save_feature_table
    from src.sequences import build_sequences, build_targets, build_vocabulary, save_sequences

    start = time.perf_counter()
    data = load_clean_data()
    features = build_patient_features(
        patients=data["patients"], diagnoses=data["diagnoses"], lab_results=data["lab_results"],
        medications=data["medications"], outcomes=data["outcomes"],
    )
    save_feature_table(features, FEATURES_PATH)
    events = build_sequences(data)
    paths = save_sequences(events, build_vocabulary(events), build_targets(data), PROCESSED, suffix=SEQUENCE_SUFFIX)
    print(f"Medications kept {len(data['medications']):,} | features {features.shape} -> {FEATURES_PATH.name} | "
          f"events {len(events):,} -> {paths['sequences'].name} | {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
