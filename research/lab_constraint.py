"""Kaggle-data results with and without the lab artifact (ISEM 735, task 23).

Task 22 found that Kaggle labs carry the lifetime diagnosis at any lead time, and that which
tests were ordered leaks it too. Part 3 feedback asked for the scheduling results with and
without the artifact. For each lab version of research/lead_time.lab_version this module
rebuilds the leak-free features, refits the calibrated XGBoost on the same splits, and re-runs
the masking frontier of research/scheduling.py:

- "panel": only the eight tests drawn at every visit (removes the ordering leak);
- "panel_w": that panel with glucose and creatinine older than the simulator's onset window W
  replaced by never-diagnosed draws (removes the lead-time artifact as well).

    python -m research.lab_constraint

This module loads XGBoost, so it must never import torch.
"""

from __future__ import annotations

import time

import pandas as pd

import run_pipeline as rp
from research.calibration import calibration_splits, fit_calibrated_xgboost, score_metrics, write_results
from research.data import features_path, load_clean_data, load_features
from research.scheduling import TARGET, Scorer, baseline_path, model_path, run_frontier
from src.features import TARGET_COLUMNS, build_patient_features, save_feature_table
from src.modeling import save_model

VERSIONS = ["panel", "panel_w"]
LABELS = {"original": "all tests (original)", "panel": "universal panel",
          "panel_w": "universal panel, constrained to W"}


def build_features(lab_version: str) -> pd.DataFrame:
    data = load_clean_data(lab_version)
    replaced = data["lab_results"].attrs.get("replaced", 0)
    print(f"{lab_version}: {len(data['lab_results']):,} lab rows, {replaced:,} replaced", flush=True)
    features = build_patient_features(
        patients=data["patients"], diagnoses=data["diagnoses"], lab_results=data["lab_results"],
        medications=data["medications"], outcomes=data["outcomes"],
    )
    save_feature_table(features, features_path(lab_version))
    return features


def check_same_patients_and_targets(features: pd.DataFrame, original: pd.DataFrame) -> None:
    """Only lab features may differ: same patients in the same order, identical targets."""
    assert features["patient_id"].tolist() == original["patient_id"].tolist()
    targets = [c for c in TARGET_COLUMNS if c in original.columns]
    pd.testing.assert_frame_equal(features[targets], original[targets], check_dtype=False)
    other = [c for c in original.columns if not c.startswith("lab_") and c in features.columns]
    pd.testing.assert_frame_equal(features[other], original[other], check_dtype=False)


def fit_and_score(features: pd.DataFrame, lab_version: str) -> dict:
    splits = calibration_splits(features, TARGET)
    X_test, y_test = splits["test"]
    _, calibrated = fit_calibrated_xgboost(splits)
    save_model(calibrated, model_path(lab_version))
    y_score = calibrated.predict_proba(X_test)[:, 1]
    pd.DataFrame({"patient_id": features.loc[X_test.index, "patient_id"], "y_true": y_test,
                  "y_score": y_score}).to_parquet(baseline_path(lab_version), index=False)
    return score_metrics(y_test, y_score)


def main() -> None:
    start = time.perf_counter()
    original = load_features()
    before = pd.read_parquet(baseline_path("original"))
    models = [{"labs": LABELS["original"], **score_metrics(before["y_true"], before["y_score"].to_numpy())}]
    frontiers = []
    for version in VERSIONS:
        features = build_features(version)
        check_same_patients_and_targets(features, original)
        models.append({"labs": LABELS[version], **fit_and_score(features, version)})
        table = run_frontier(Scorer(lab_version=version), lab_version=version)
        rp.save_metrics(table, rp.TABLES_DIR / f"scheduling_frontier_masking_{version}.csv")
        frontiers.append(table.assign(labs=LABELS[version]))
        print(f"{version} done | {time.perf_counter() - start:.0f}s", flush=True)

    models = pd.DataFrame(models)
    print(models[["labs", "roc_auc", "brier", "calibration_slope"]].round(4).to_string(index=False))
    frontier = pd.concat(frontiers, ignore_index=True)
    show = ["labs", "controller", "setting", "share_of_observed", "roc_auc", "brier"]
    print(frontier[show].round(4).to_string(index=False))

    log = pd.concat([
        models.assign(target=TARGET, model="calibrated XGBoost, " + models["labs"]),
        frontier.assign(target=TARGET, model="masking, " + frontier["labs"] + ": " + frontier["controller"] + " "
                        + frontier["setting"].astype(str)),
    ], ignore_index=True)
    write_results(log, task="23", seed=rp.RANDOM_STATE, metrics=["roc_auc", "brier"],
                  notes="leak-free data; universal 8-test panel, then glucose/creatinine older than simulator "
                        "window W replaced by never-diagnosed draws; same splits; random = mean of 5 seeds")
    print(f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
