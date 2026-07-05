"""Simple pipeline runner for the disease progression project.

This script regenerates the key project artifacts without opening the
notebooks. It reuses the helper functions in src/ and follows the same
methodology as the notebooks:

- future_hospitalization uses all patients.
- multimorbidity_progression uses only the "at-risk" cohort of patients with
  fewer than two baseline conditions (matching notebook 07).
- Both use a stratified 75/25 train/test split.

Usage:
    python run_pipeline.py --step features
    python run_pipeline.py --step models
    python run_pipeline.py --step evaluate
    python run_pipeline.py --step calibrate   # threshold + calibration analysis
    python run_pipeline.py --step explain     # SHAP summary + top-features table
    python run_pipeline.py --step survival    # Kaplan-Meier + Cox progression model
    python run_pipeline.py --step all

    # Faster run on a random patient sample (useful for smoke tests):
    python run_pipeline.py --step all --sample 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_cleaning import clean_all, load_raw_data
from src.evaluation import (
    evaluate_models,
    plot_calibration_curve,
    plot_precision_recall_curve,
    plot_roc_curves,
    predict_scores,
    save_metrics,
    threshold_table,
)
from src.data_cleaning import make_condition_events
from src.features import build_patient_features, get_modeling_matrices, save_feature_table
from src.modeling import fit_models, load_model, make_advanced_models, make_baseline_models, save_model
from src.survival import build_time_to_progression, fit_cox_model, plot_km_by_condition_group
from src.visualization import plot_feature_importance, plot_model_comparison, save_shap_summary

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"
CONDITION_EVENTS_PATH = PROJECT_ROOT / "data" / "processed" / "condition_events.csv"
MODELS_DIR = PROJECT_ROOT / "models"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# The two main prediction targets used throughout the project.
TARGETS = ["future_hospitalization", "multimorbidity_progression"]
RANDOM_STATE = 42
TEST_SIZE = 0.25


def load_feature_table(sample: int | None = None) -> pd.DataFrame:
    """Load the saved modeling dataset, optionally sampling patients."""
    if not FEATURE_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURE_TABLE_PATH} not found. Run: python run_pipeline.py --step features"
        )
    features = pd.read_csv(FEATURE_TABLE_PATH)
    if sample is not None and sample < len(features):
        features = features.sample(n=sample, random_state=RANDOM_STATE).reset_index(drop=True)
    return features


def cohort_for_target(features: pd.DataFrame, target: str) -> pd.DataFrame:
    """Restrict to the modeling cohort for a target, matching the notebooks.

    Multimorbidity progression is only defined for patients who start with fewer
    than two baseline conditions, so we predict it within that at-risk cohort.
    """
    if target == "multimorbidity_progression":
        return features[features["baseline_condition_count"] < 2].copy()
    return features


def split_for_target(features: pd.DataFrame, target: str):
    """Create the same stratified train/test split used in the notebooks."""
    cohort = cohort_for_target(features, target)
    X, y = get_modeling_matrices(cohort, target=target)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def _pos_weight(y_train: pd.Series) -> float | None:
    """Ratio of negatives to positives, used for XGBoost class balancing."""
    positives = int(y_train.sum())
    if positives == 0:
        return None
    return float((len(y_train) - positives) / positives)


def step_features() -> None:
    """Rebuild the patient-level modeling dataset from the raw CSVs."""
    print("Loading and cleaning raw data...")
    data = clean_all(load_raw_data())
    print("Building patient features (cutoff 2021-12-31)...")
    features = build_patient_features(
        patients=data["patients"],
        diagnoses=data["diagnoses"],
        lab_results=data["lab_results"],
        medications=data["medications"],
        outcomes=data["outcomes"],
    )
    path = save_feature_table(features, FEATURE_TABLE_PATH)
    print(f"Saved {len(features):,} rows x {features.shape[1]} columns to {path}")


def step_models(sample: int | None = None) -> None:
    """Train baseline and advanced models for each target and save them."""
    features = load_feature_table(sample)
    for target in TARGETS:
        print(f"\n=== Training models for {target} ===")
        X_train, X_test, y_train, _ = split_for_target(features, target)
        models = {
            **make_baseline_models(X_train),
            **make_advanced_models(X_train, random_state=RANDOM_STATE, pos_weight=_pos_weight(y_train)),
        }
        print(
            f"Cohort {len(X_train) + len(X_test):,} patients "
            f"(train {len(X_train):,}) | positives {int(y_train.sum()):,} "
            f"| models: {', '.join(models)}"
        )
        fitted = fit_models(models, X_train, y_train)
        for name, model in fitted.items():
            path = save_model(model, MODELS_DIR / f"{target}_{name}.joblib")
            print(f"Saved {path.name}")


def step_evaluate(sample: int | None = None) -> None:
    """Evaluate saved models, save the combined metrics tables and figures."""
    features = load_feature_table(sample)
    all_metrics = []
    for target in TARGETS:
        _, X_test, _, y_test = split_for_target(features, target)
        model_paths = sorted(MODELS_DIR.glob(f"{target}_*.joblib"))
        if not model_paths:
            print(f"No saved models found for {target}; skipping.")
            continue
        models = {}
        for path in model_paths:
            try:
                models[path.stem.replace(f"{target}_", "")] = load_model(path)
            except Exception as error:
                print(
                    f"Could not load {path.name} ({type(error).__name__}). "
                    "It may have been saved with a different scikit-learn version; "
                    "retrain with: python run_pipeline.py --step models"
                )
        if not models:
            continue
        metrics = evaluate_models(models, X_test, y_test, target_name=target)
        all_metrics.append(metrics)
        print(f"\n=== Metrics for {target} ===")
        print(metrics.to_string(index=False))

        plot_roc_curves(
            models,
            X_test,
            y_test,
            title=f"ROC Curves: {target}",
            path=FIGURES_DIR / f"{target}_roc_curves.png",
        )

    if not all_metrics:
        return

    combined = pd.concat(all_metrics, ignore_index=True)
    combined = combined.sort_values(["target", "roc_auc", "f1"], ascending=[True, False, False])
    best_by_target = combined.groupby("target").head(1).reset_index(drop=True)

    save_metrics(combined, TABLES_DIR / "08_all_model_metrics.csv")
    save_metrics(best_by_target, TABLES_DIR / "08_best_models_by_target.csv")
    plot_model_comparison(combined, metric="roc_auc", path=FIGURES_DIR / "08_model_comparison_roc_auc.png")
    plot_model_comparison(combined, metric="f1", path=FIGURES_DIR / "08_model_comparison_f1.png")
    print(f"\nSaved combined metrics and comparison figures to {TABLES_DIR} and {FIGURES_DIR}")


# Tree models, best first, that TreeExplainer can explain with SHAP.
TREE_MODELS = ["xgboost", "lightgbm", "random_forest"]


def _best_model_name(target: str, model_names: list[str]) -> str:
    """Pick the best non-trivial model for a target from the saved metrics."""
    best_table = TABLES_DIR / "08_best_models_by_target.csv"
    if best_table.exists():
        best = pd.read_csv(best_table)
        match = best[best["target"] == target]
        if not match.empty and match.iloc[0]["model"] in model_names:
            return str(match.iloc[0]["model"])
    # Fall back to a real classifier rather than the dummy baseline.
    for preferred in ["xgboost", "lightgbm", "random_forest", "logistic_regression"]:
        if preferred in model_names:
            return preferred
    return model_names[0]


def _best_tree_model_name(target: str, available: dict) -> str | None:
    """Pick the best-ranked tree model for a target (SHAP needs a tree model)."""
    best = _best_model_name(target, list(available))
    if best in TREE_MODELS:
        return best
    for name in TREE_MODELS:
        if name in available:
            return name
    return None


def step_explain(sample: int | None = None, shap_sample: int = 500) -> None:
    """Explain the best tree model for each target with SHAP (feature-importance fallback)."""
    features = load_feature_table(sample)
    for target in TARGETS:
        _, X_test, _, _ = split_for_target(features, target)
        model_paths = sorted(MODELS_DIR.glob(f"{target}_*.joblib"))
        available = {path.stem.replace(f"{target}_", ""): path for path in model_paths}
        best_name = _best_tree_model_name(target, available)
        if best_name is None:
            print(f"No tree model available for {target}; skipping explainability.")
            continue

        model = load_model(available[best_name])
        X_shap = X_test.head(shap_sample)
        print(f"\n=== Explaining {target} using '{best_name}' on {len(X_shap)} rows ===")

        try:
            top = save_shap_summary(
                model,
                X_shap,
                title=f"SHAP summary: {target} ({best_name})",
                path=FIGURES_DIR / f"shap_summary_{target}.png",
            )
            save_metrics(top, TABLES_DIR / f"shap_top_features_{target}.csv")
            print("SHAP summary saved. Top 5 features by mean |SHAP|:")
            print(top.head(5).to_string(index=False))
        except Exception as error:
            print(f"SHAP unavailable ({type(error).__name__}: {error}); using feature importance instead.")
            try:
                _, _, importance = plot_feature_importance(
                    model, top_n=20, path=FIGURES_DIR / f"feature_importance_{target}.png"
                )
                save_metrics(importance.head(20), TABLES_DIR / f"feature_importance_{target}.csv")
                print("Feature importance saved. Top 5 features:")
                print(importance.head(5).to_string(index=False))
            except Exception as fallback_error:
                print(f"Feature importance also unavailable: {fallback_error}")


def step_calibrate(target: str = "future_hospitalization", sample: int | None = None) -> None:
    """Threshold, calibration, and precision-recall analysis for the best model.

    Care teams act on a risk threshold, not a model ranking, so this reports the
    precision/recall/F1 trade-off across thresholds, a reliability diagram with
    the Brier score, and the precision-recall curve.
    """
    features = load_feature_table(sample)
    _, X_test, _, y_test = split_for_target(features, target)

    model_paths = sorted(MODELS_DIR.glob(f"{target}_*.joblib"))
    available = {path.stem.replace(f"{target}_", ""): path for path in model_paths}
    if not available:
        raise FileNotFoundError(f"No saved models for {target}; run --step models first.")

    best_name = _best_model_name(target, list(available))
    print(f"Calibration/threshold analysis for {target} using '{best_name}'...")
    model = load_model(available[best_name])
    y_score = predict_scores(model, X_test)

    table = threshold_table(y_test, y_score)
    save_metrics(table, TABLES_DIR / "threshold_analysis.csv")

    _, _, brier = plot_calibration_curve(
        y_test,
        y_score,
        title=f"Calibration: {target} ({best_name})",
        path=FIGURES_DIR / "calibration_curve.png",
    )
    _, _, ap = plot_precision_recall_curve(
        y_test,
        y_score,
        title=f"Precision-Recall: {target} ({best_name})",
        path=FIGURES_DIR / "precision_recall_curve.png",
    )

    print(f"Brier score: {brier:.4f} | Average precision: {ap:.4f}")
    print("\nThreshold trade-offs (subset):")
    subset = table[table["threshold"].isin([0.1, 0.2, 0.3, 0.5, 0.7])]
    print(subset.to_string(index=False))
    print(
        f"\nSaved threshold_analysis.csv, calibration_curve.png, and "
        f"precision_recall_curve.png under outputs/."
    )


def _load_condition_events() -> pd.DataFrame:
    """Load the processed condition-event table, rebuilding it from raw if absent."""
    if CONDITION_EVENTS_PATH.exists():
        return pd.read_csv(CONDITION_EVENTS_PATH)
    print("condition_events.csv not found; rebuilding from raw diagnoses...")
    data = clean_all(load_raw_data())
    return make_condition_events(data["diagnoses"])


def step_survival(sample: int | None = None) -> None:
    """Time-to-multimorbidity-progression analysis: Kaplan-Meier and Cox model."""
    features = load_feature_table(sample)
    condition_events = _load_condition_events()

    survival_df = build_time_to_progression(condition_events, features)
    events = int(survival_df["event_observed"].sum())
    print(
        f"At-risk cohort: {len(survival_df):,} patients | "
        f"progression events: {events:,} ({events / len(survival_df):.1%}) | "
        f"median follow-up: {survival_df['duration_days'].median():.0f} days"
    )

    plot_km_by_condition_group(survival_df, path=FIGURES_DIR / "km_curve_progression.png")
    summary = fit_cox_model(survival_df, path=TABLES_DIR / "cox_model_summary.csv")

    print("\nCox proportional hazards (hazard ratios):")
    print(
        summary[["covariate", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
        .to_string(index=False)
    )
    print("\nSaved km_curve_progression.png and cox_model_summary.csv under outputs/.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate key project artifacts.")
    parser.add_argument(
        "--step",
        choices=["features", "models", "evaluate", "calibrate", "explain", "survival", "all"],
        required=True,
        help="Which pipeline step to run.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional number of patients to sample for faster model runs.",
    )
    args = parser.parse_args()

    if args.step in ("features", "all"):
        step_features()
    if args.step in ("models", "all"):
        step_models(args.sample)
    if args.step in ("evaluate", "all"):
        step_evaluate(args.sample)
    if args.step in ("calibrate", "all"):
        step_calibrate(sample=args.sample)
    if args.step in ("explain", "all"):
        step_explain(sample=args.sample)
    if args.step in ("survival", "all"):
        step_survival(sample=args.sample)


if __name__ == "__main__":
    main()
