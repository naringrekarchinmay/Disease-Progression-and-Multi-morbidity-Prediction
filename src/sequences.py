"""Per-patient event sequences for the ISEM 735 sequence models.

Each patient becomes one time-ordered stream of events built from the five raw
tables, using only events on or before the cutoff date.

Build and save (a sample first, then everything):
    python -m src.sequences --sample 5000
    python -m src.sequences
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_cleaning import clean_all, load_raw_data
from src.features import TARGET_COLUMNS, build_patient_features

EVENT_COLUMNS = ["patient_id", "date", "event_type", "token", "days_since_prev", "age_at_event"]
# Same-day events are ordered by type, then token, so the stream is deterministic.
EVENT_TYPE_ORDER = {"diagnosis": 0, "lab": 1, "medication": 2, "admission": 3}
# Reserved token ids: 0 pads short sequences, 1 stands in for tokens unseen in training.
PAD_ID, UNK_ID = 0, 1
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RANDOM_STATE = 42


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



def build_vocabulary(events: pd.DataFrame) -> pd.DataFrame:
    """Map each token to an integer id, most frequent first, after the reserved ids."""
    counts = events["token"].value_counts().rename_axis("token").reset_index(name="count")
    counts = counts.sort_values(["count", "token"], ascending=[False, True], kind="stable").reset_index(drop=True)
    counts.insert(1, "token_id", range(UNK_ID + 1, UNK_ID + 1 + len(counts)))
    return counts

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


def save_sequences(
    events: pd.DataFrame,
    vocabulary: pd.DataFrame,
    targets: pd.DataFrame,
    out_dir: Path | str = PROCESSED_DIR,
    suffix: str = "",
) -> dict[str, Path]:
    """Write the event stream (with token ids), vocabulary, and targets to `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sequences": out_dir / f"sequences{suffix}.parquet",
        "vocabulary": out_dir / f"sequences_vocab{suffix}.csv",
        "targets": out_dir / f"sequence_targets{suffix}.parquet",
    }
    token_ids = events["token"].map(vocabulary.set_index("token")["token_id"])
    events.assign(token_id=token_ids.fillna(UNK_ID).astype(int)).to_parquet(paths["sequences"], index=False)
    vocabulary.to_csv(paths["vocabulary"], index=False)
    targets.to_parquet(paths["targets"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-patient event sequences.")
    parser.add_argument("--sample", type=int, default=None, help="Build for a random sample of N patients.")
    args = parser.parse_args()

    start = time.perf_counter()
    raw = load_raw_data()
    if args.sample is not None:
        ids = set(raw["patients"]["patient_id"].sample(args.sample, random_state=RANDOM_STATE))
        raw = {name: table[table["patient_id"].isin(ids)] for name, table in raw.items()}
    data = clean_all(raw)
    loaded = time.perf_counter()

    events = build_sequences(data)
    vocabulary = build_vocabulary(events)
    targets = build_targets(data)
    built = time.perf_counter()

    suffix = f"_sample{args.sample}" if args.sample is not None else ""
    paths = save_sequences(events, vocabulary, targets, suffix=suffix)
    done = time.perf_counter()

    per_patient = events.groupby("patient_id").size()
    print(f"Patients: {len(targets):,} | with input events: {per_patient.size:,}")
    print(f"Events: {len(events):,} | per patient median {per_patient.median():.0f}, max {per_patient.max()}")
    print(f"Vocabulary: {len(vocabulary)} tokens (+ PAD and UNK)")
    print(f"Runtime: load {loaded - start:.1f}s, build {built - loaded:.1f}s, save {done - built:.1f}s, total {done - start:.1f}s")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
