"""Feature engineering for disease progression and multimorbidity prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_cleaning import make_condition_events


TARGET_COLUMNS = [
    "future_hospitalization",
    "future_icu_admission",
    "future_30d_readmission",
    "multimorbidity_progression",
    "new_condition_count",
    "future_condition_count",
]


def _safe_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Make generated feature names friendly for CSVs and model pipelines."""
    result = df.copy()
    result.columns = [
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("|", "_")
        for col in result.columns
    ]
    return result


def _prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [f"{prefix}_{col}" for col in renamed.columns]
    return renamed


def build_patient_features(
    patients: pd.DataFrame,
    diagnoses: pd.DataFrame,
    lab_results: pd.DataFrame,
    medications: pd.DataFrame,
    outcomes: pd.DataFrame,
    cutoff_date: str = "2021-12-31",
    horizon_end: str = "2024-12-31",
) -> pd.DataFrame:
    """Build one modeling row per patient using only pre-cutoff features.

    Targets are created from events after the cutoff date through the horizon end.
    The patient-level dx_* columns are excluded from predictors because they may
    represent all-time diagnoses rather than information known at the cutoff.
    """
    cutoff = pd.Timestamp(cutoff_date)
    horizon = pd.Timestamp(horizon_end)

    base = patients.copy()
    base["patient_id"] = base["patient_id"].astype(str)
    dx_cols = [col for col in base.columns if col.startswith("dx_")]
    base = base.drop(columns=dx_cols, errors="ignore")

    condition_events = make_condition_events(diagnoses)
    baseline_conditions = condition_events[condition_events["visit_date"] <= cutoff].copy()
    baseline_condition_counts = (
        baseline_conditions.groupby("patient_id")["condition"].nunique().rename("baseline_condition_count")
    )
    all_future_condition_counts = (
        condition_events[condition_events["visit_date"] <= horizon]
        .groupby("patient_id")["condition"]
        .nunique()
        .rename("future_condition_count")
    )

    visit_summary = baseline_conditions.groupby("patient_id").agg(
        baseline_visit_count=("visit_date", "count"),
        baseline_visit_type_count=("visit_type", "nunique"),
        baseline_specialty_count=("provider_specialty", "nunique"),
    )

    condition_flags = pd.crosstab(baseline_conditions["patient_id"], baseline_conditions["condition"])
    condition_flags = _prefix_columns(_safe_column_names(condition_flags), "baseline_dx")
    condition_flags = (condition_flags > 0).astype(int)

    labs = lab_results.copy()
    labs["test_date"] = pd.to_datetime(labs["test_date"], errors="coerce")
    baseline_labs = labs[labs["test_date"] <= cutoff].copy()
    lab_summary = baseline_labs.groupby("patient_id").agg(
        lab_test_count=("test_name", "count"),
        lab_unique_test_count=("test_name", "nunique"),
        lab_abnormal_rate=("is_abnormal", "mean"),
        lab_mean_delta_from_normal=("delta_from_normal", "mean"),
    )
    lab_value_summary = baseline_labs.pivot_table(
        index="patient_id",
        columns="test_name",
        values="value",
        aggfunc=["mean", "max"],
    )
    lab_value_summary.columns = [f"lab_{agg}_{test}" for agg, test in lab_value_summary.columns]
    lab_value_summary = _safe_column_names(lab_value_summary)

    meds = medications.copy()
    meds["start_date"] = pd.to_datetime(meds["start_date"], errors="coerce")
    baseline_meds = meds[meds["start_date"] <= cutoff].copy()
    med_summary = baseline_meds.groupby("patient_id").agg(
        medication_count=("medication", "count"),
        medication_unique_count=("medication", "nunique"),
        medication_indication_count=("indication", "nunique"),
        medication_mean_adherence=("adherence_pct", "mean"),
        medication_mean_duration_days=("duration_days", "mean"),
        medication_generic_rate=("is_generic", "mean"),
    )
    med_indication_flags = pd.crosstab(baseline_meds["patient_id"], baseline_meds["indication"])
    med_indication_flags = _prefix_columns(_safe_column_names(med_indication_flags), "med_indication")
    med_indication_flags = (med_indication_flags > 0).astype(int)

    hospital = outcomes.copy()
    hospital["admission_date"] = pd.to_datetime(hospital["admission_date"], errors="coerce")
    prior_hospital = hospital[hospital["admission_date"] <= cutoff].copy()
    future_hospital = hospital[
        (hospital["admission_date"] > cutoff) & (hospital["admission_date"] <= horizon)
    ].copy()

    prior_hospital_summary = prior_hospital.groupby("patient_id").agg(
        prior_hospitalization_count=("admission_date", "count"),
        prior_icu_admission_count=("icu_admission", "sum"),
        prior_mean_los=("length_of_stay_days", "mean"),
        prior_readmission_rate=("readmitted_30d", "mean"),
    )

    target_summary = future_hospital.groupby("patient_id").agg(
        future_hospitalization=("admission_date", lambda s: int(s.notna().any())),
        future_icu_admission=("icu_admission", "max"),
        future_30d_readmission=("readmitted_30d", "max"),
    )

    features = base.set_index("patient_id")
    for table in [
        baseline_condition_counts,
        visit_summary,
        condition_flags,
        lab_summary,
        lab_value_summary,
        med_summary,
        med_indication_flags,
        prior_hospital_summary,
        target_summary,
        all_future_condition_counts,
    ]:
        features = features.join(table, how="left")

    count_cols = [
        "baseline_condition_count",
        "baseline_visit_count",
        "baseline_visit_type_count",
        "baseline_specialty_count",
        "lab_test_count",
        "lab_unique_test_count",
        "medication_count",
        "medication_unique_count",
        "medication_indication_count",
        "prior_hospitalization_count",
        "prior_icu_admission_count",
        "future_condition_count",
    ]
    for col in count_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    binary_targets = ["future_hospitalization", "future_icu_admission", "future_30d_readmission"]
    for col in binary_targets:
        if col in features.columns:
            features[col] = features[col].fillna(0).astype(int)
        else:
            features[col] = 0

    features["new_condition_count"] = (
        features["future_condition_count"] - features["baseline_condition_count"]
    ).clip(lower=0)
    features["multimorbidity_progression"] = (
        (features["baseline_condition_count"] < 2) & (features["future_condition_count"] >= 2)
    ).astype(int)

    rate_cols = [col for col in features.columns if col.endswith("_rate")]
    features[rate_cols] = features[rate_cols].fillna(0)
    features = features.reset_index()
    return _safe_column_names(features)


def get_modeling_matrices(
    feature_df: pd.DataFrame,
    target: str,
    drop_targets: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return X and y for a selected target."""
    if target not in feature_df.columns:
        raise ValueError(f"Target {target!r} was not found in the feature table.")

    drop_cols = ["patient_id"]
    if drop_targets:
        drop_cols.extend([col for col in TARGET_COLUMNS if col in feature_df.columns])

    X = feature_df.drop(columns=drop_cols, errors="ignore")
    y = feature_df[target].astype(int)
    return X, y


def save_feature_table(feature_df: pd.DataFrame, path: str | Path) -> Path:
    """Save the modeling dataset to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    return output_path
