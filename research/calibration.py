"""Isotonic calibration of the XGBoost baselines (ISEM 735, Task 6).

The saved baselines were trained on the whole training split, so calibrating them
on part of it would be optimistic. Instead, a validation split is carved from the
training split, XGBoost is refitted on the rest with the same settings, and an
isotonic calibrator is fitted on the validation split. The test split is the
pipeline's own and is used only for the final scores.

    python -m research.calibration
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import run_pipeline as rp
from src.evaluation import predict_scores
from src.modeling import load_model, make_advanced_models, save_model

VALIDATION_SIZE = 0.2
RESULTS_LOG = Path(__file__).resolve().parent / "results_log.csv"


def calibration_splits(
    features: pd.DataFrame,
    target: str,
    validation_size: float = VALIDATION_SIZE,
    seed: int = rp.RANDOM_STATE,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Return (X, y) for fit, validation and test.

    The test split is exactly `run_pipeline.split_for_target`'s; validation is a
    stratified share of that function's training split.
    """
    X_train, X_test, y_train, y_test = rp.split_for_target(features, target)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=validation_size, random_state=seed, stratify=y_train
    )
    return {"fit": (X_fit, y_fit), "validation": (X_val, y_val), "test": (X_test, y_test)}


def fit_calibrated_xgboost(
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
    seed: int = rp.RANDOM_STATE,
) -> tuple[Pipeline, CalibratedClassifierCV]:
    """Refit the baseline XGBoost on the fit split, then calibrate it on validation.

    Only the fit and validation splits are read; the test split is never touched.
    """
    X_fit, y_fit = splits["fit"]
    X_val, y_val = splits["validation"]
    base = make_advanced_models(X_fit, random_state=seed, pos_weight=rp._pos_weight(y_fit))["xgboost"]
    base.fit(X_fit, y_fit)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    calibrated.fit(X_val, y_val)
    return base, calibrated


def calibration_slope(y_true: pd.Series, y_score: np.ndarray, eps: float = 1e-6) -> float:
    """Slope of a logistic fit of the outcome on logit(predicted risk); 1 is ideal."""
    p = np.clip(y_score, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return float(LogisticRegression(C=np.inf).fit(logit, y_true).coef_[0, 0])


def score_metrics(y_true: pd.Series, y_score: np.ndarray) -> dict[str, float]:
    brier = brier_score_loss(y_true, y_score)
    rate = float(np.mean(y_true))
    return {
        "brier": brier,
        # Share of the no-skill Brier (always predicting the base rate) that the model removes.
        "brier_skill_score": 1 - brier / (rate * (1 - rate)),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "calibration_slope": calibration_slope(y_true, y_score),
        "mean_predicted": float(np.mean(y_score)),
        "observed_rate": float(np.mean(y_true)),
    }


def plot_before_after(y_true: pd.Series, scores: dict[str, np.ndarray], target: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfectly calibrated")
    for (name, y_score), color in zip(scores.items(), ["#C44E52", "#DD8452", "#4C72B0"]):
        prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", color=color,
                label=f"{name} (Brier {brier_score_loss(y_true, y_score):.3f})")
    ax.set_xlabel("Mean predicted risk (test-set deciles)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration before and after: {target}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    seed = rp.RANDOM_STATE
    features = rp.load_feature_table()
    rows = []
    for target in rp.TARGETS:
        start = time.perf_counter()
        splits = calibration_splits(features, target, seed=seed)
        X_test, y_test = splits["test"]
        base, calibrated = fit_calibrated_xgboost(splits, seed=seed)
        save_model(calibrated, rp.MODELS_DIR / f"{target}_xgboost_isotonic.joblib")

        scores = {
            "original XGBoost": predict_scores(load_model(rp.MODELS_DIR / f"{target}_xgboost.joblib"), X_test),
            "refit XGBoost (80% of train)": predict_scores(base, X_test),
            "refit + isotonic": calibrated.predict_proba(X_test)[:, 1],
        }
        plot_before_after(y_test, scores, target, rp.FIGURES_DIR / f"calibration_before_after_{target}.png")
        for model, y_score in scores.items():
            rows.append({"target": target, "model": model, **score_metrics(y_test, y_score)})
        sizes = {name: len(split[1]) for name, split in splits.items()}
        print(f"{target}: fit/validation/test = {sizes} in {time.perf_counter() - start:.1f}s")

    table = pd.DataFrame(rows)
    rp.save_metrics(table, rp.TABLES_DIR / "calibration_before_after.csv")
    print(table.round(4).to_string(index=False))

    today = dt.date.today().isoformat()
    log = table.melt(id_vars=["target", "model"], var_name="metric", value_name="value")
    log = log[log["metric"].isin(["brier", "brier_skill_score", "roc_auc", "average_precision", "calibration_slope"])]
    log = log.assign(date=today, task="6", seed=seed,
                     notes="test split of run_pipeline; isotonic fitted on 20% validation carved from train")
    log[["date", "task", "model", "target", "metric", "value", "seed", "notes"]].to_csv(
        RESULTS_LOG, mode="a", header=False, index=False
    )


if __name__ == "__main__":
    main()
