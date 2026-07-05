"""Plotting helpers for EDA and model interpretation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FIGURES_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"


def set_plot_style() -> None:
    """Apply a clean plotting style used across notebooks."""
    sns.set_theme(style="whitegrid", palette="Set2")
    plt.rcParams["figure.dpi"] = 120


def save_fig(fig, filename: str, output_dir: Path | str = FIGURES_DIR) -> Path:
    """Save a matplotlib figure under outputs/figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_multimorbidity_distribution(feature_df: pd.DataFrame, path: str | Path | None = None):
    """Plot the baseline multimorbidity burden distribution."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=feature_df, x="baseline_condition_count", ax=ax, color="#4C78A8")
    ax.set_title("Baseline Condition Count per Patient")
    ax.set_xlabel("Number of unique baseline conditions")
    ax.set_ylabel("Patient count")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_condition_prevalence(condition_events: pd.DataFrame, top_n: int = 15, path: str | Path | None = None):
    """Plot the most common diagnosis conditions."""
    set_plot_style()
    counts = condition_events["condition"].value_counts().head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    counts.plot(kind="barh", ax=ax, color="#59A14F")
    ax.set_title(f"Top {top_n} Diagnosis Conditions")
    ax.set_xlabel("Diagnosis event count")
    ax.set_ylabel("")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_model_comparison(metrics_df: pd.DataFrame, metric: str = "roc_auc", path: str | Path | None = None):
    """Plot model performance for one metric."""
    set_plot_style()
    plot_df = metrics_df.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=plot_df, x=metric, y="model", hue="target", dodge=False, ax=ax)
    ax.set_title(f"Model Comparison by {metric.upper()}")
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Model")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig, ax


def get_feature_names_from_pipeline(model) -> list[str]:
    """Extract transformed feature names from a fitted sklearn pipeline."""
    preprocessor = model.named_steps.get("preprocess")
    if preprocessor is None:
        return []
    return list(preprocessor.get_feature_names_out())


def compute_shap_values(model, X_sample: pd.DataFrame):
    """Return (shap_values_2d, transformed_matrix, feature_names) for a tree pipeline.

    Handles the different shapes SHAP returns across versions and estimators:
    a list per class, a 3D array (samples, features, classes), or a plain 2D
    array. The result is always the 2D positive-class contribution matrix.
    """
    import numpy as np
    import shap

    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    transformed = preprocessor.transform(X_sample)
    feature_names = get_feature_names_from_pipeline(model)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[-1])  # positive class
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, -1]  # (samples, features, classes) -> positive class
    return values, transformed, feature_names


def shap_top_features(shap_values, feature_names: list[str], top_n: int = 20) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value (average impact on predictions)."""
    import numpy as np

    mean_abs = np.abs(shap_values).mean(axis=0)
    table = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True).head(top_n)


def save_shap_summary(model, X_sample: pd.DataFrame, title: str, path: str | Path, top_n: int = 20):
    """Save a SHAP beeswarm summary plot and return the top-features table.

    Requires the optional `shap` package; the caller should fall back to plain
    feature importance if this raises.
    """
    import shap

    values, transformed, feature_names = compute_shap_values(model, X_sample)

    plt.figure()
    shap.summary_plot(values, transformed, feature_names=feature_names, show=False, max_display=top_n)
    fig = plt.gcf()
    fig.suptitle(title, y=1.02)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return shap_top_features(values, feature_names, top_n=top_n)


def plot_feature_importance(model, top_n: int = 20, path: str | Path | None = None):
    """Plot model feature importances when the final estimator exposes them."""
    estimator = model.named_steps.get("model")
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError("This model does not expose feature_importances_.")

    names = get_feature_names_from_pipeline(model)
    importance = pd.DataFrame(
        {"feature": names, "importance": estimator.feature_importances_}
    ).sort_values("importance", ascending=False)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=importance.head(top_n), x="importance", y="feature", ax=ax, color="#F28E2B")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig, ax, importance

