"""Evaluation helpers for classification models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)


def predict_scores(model, X: pd.DataFrame) -> pd.Series:
    """Return positive-class scores from a fitted classifier."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def classification_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calculate several metrics for a binary classifier."""
    y_pred = model.predict(X_test)
    y_score = predict_scores(model, X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_test.nunique() > 1:
        metrics["roc_auc"] = roc_auc_score(y_test, y_score)
        metrics["average_precision"] = average_precision_score(y_test, y_score)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    return metrics


def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series, target_name: str) -> pd.DataFrame:
    """Evaluate fitted models and return a tidy metrics table."""
    rows = []
    for name, model in models.items():
        row = {"model": name, "target": target_name}
        row.update(classification_metrics(model, X_test, y_test))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "roc_auc", "f1"], ascending=[True, False, False])


def save_metrics(metrics_df: pd.DataFrame, path: str | Path) -> Path:
    """Save model metrics to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    return output_path


def threshold_table(
    y_true: pd.Series,
    y_score: pd.Series,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Precision, recall, F1 and flag rate at several decision thresholds.

    Care teams pick an operating threshold rather than using a fixed 0.5, so
    this table shows how many patients get flagged and how the trade-off shifts.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    positives = int(y_true.sum())

    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        flagged = int(y_pred.sum())
        rows.append(
            {
                "threshold": threshold,
                "n_flagged": flagged,
                "flag_rate": flagged / len(y_true),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "true_positives": int(((y_pred == 1) & (y_true == 1)).sum()),
                "of_actual_positives": positives,
            }
        )
    return pd.DataFrame(rows)


def plot_calibration_curve(
    y_true: pd.Series,
    y_score: pd.Series,
    title: str,
    n_bins: int = 10,
    path: str | Path | None = None,
):
    """Plot a reliability diagram comparing predicted vs observed risk."""
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, marker="o", label=f"Model (Brier = {brier:.4f})")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig, ax, brier


def plot_precision_recall_curve(
    y_true: pd.Series,
    y_score: pd.Series,
    title: str,
    path: str | Path | None = None,
):
    """Plot the precision-recall curve, useful for imbalanced targets."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = float(np.asarray(y_true).mean())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"Model (AP = {ap:.3f})")
    ax.axhline(baseline, linestyle="--", color="gray", linewidth=1, label=f"Baseline rate = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig, ax, ap


def plot_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, title: str, path: str | Path | None = None):
    """Plot and optionally save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_roc_curves(models: dict, X_test: pd.DataFrame, y_test: pd.Series, title: str, path: str | Path | None = None):
    """Plot ROC curves for fitted models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.set_title(title)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig, ax

