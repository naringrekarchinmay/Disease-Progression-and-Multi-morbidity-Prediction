"""Monte Carlo dropout uncertainty for the time-decay GRU (ISEM 735, Task 8).

Dropout stays on at prediction time and each test patient is scored 30 times; the
spread of those predictions is the model's uncertainty about that patient.

    python -m research.uncertainty

Like research.gru_model, this module must never load XGBoost (see that module).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import run_pipeline as rp
from research.calibration import write_results
from research.gru_model import (
    MODEL_PATH,
    SEQUENCES_PATH,
    TARGET,
    TimeDecayGRU,
    predict_proba,
    sequence_splits,
    split_tensors,
)

PASSES = 30
PREDICTIONS_PATH = rp.PROJECT_ROOT / "data" / "processed" / "gru_mc_dropout_test.parquet"


def mc_dropout_predict(
    model: TimeDecayGRU,
    tensors: dict[str, torch.Tensor],
    passes: int = PASSES,
    seed: int = 42,
    batch_size: int = 2048,
) -> np.ndarray:
    """Predicted probabilities with dropout active, shape [passes, patients].

    The model is returned to evaluation mode afterwards.
    """
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    model.train()  # the GRU has dropout but no batch norm, so train() only switches dropout on
    try:
        with torch.no_grad():
            draws = [
                torch.cat(
                    [
                        torch.sigmoid(model({k: v[i : i + batch_size] for k, v in tensors.items()}))
                        for i in range(0, len(tensors["lengths"]), batch_size)
                    ]
                )
                for _ in range(passes)
            ]
    finally:
        model.eval()
        torch.random.set_rng_state(generator_state)
    return torch.stack(draws).numpy()


def lab_visits_before_cutoff(events: pd.DataFrame, ids: list[str]) -> np.ndarray:
    """Distinct lab-visit days per patient, in `ids` order (events are already pre-cutoff)."""
    labs = events[events["event_type"] == "lab"]
    per_patient = labs.groupby("patient_id")["date"].nunique()
    return per_patient.reindex(ids, fill_value=0).to_numpy()


def selective_auroc(y: np.ndarray, score: np.ndarray, uncertainty: np.ndarray, keep: float) -> float:
    """AUROC on the `keep` share of patients the model is most certain about."""
    order = np.argsort(uncertainty, kind="stable")[: int(round(keep * len(y)))]
    return roc_auc_score(y[order], score[order])


def within_decile_spearman(df: pd.DataFrame) -> float:
    """Mean Spearman of uncertainty against error inside each decile of predicted risk.

    Near 0 means dropout spread says nothing about error beyond the risk score itself.
    """
    deciles = pd.qcut(df["mc_mean"], 10, labels=False)
    return float(np.mean([spearmanr(g["mc_std"], g["abs_error"]).statistic for _, g in df.groupby(deciles)]))


def plot_uncertainty(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for outcome, color in [(0, "#4C72B0"), (1, "#C44E52")]:
        part = df[df["y_true"] == outcome]
        ax.scatter(part["mc_mean"], part["mc_std"], s=4, alpha=0.25, color=color,
                   label=f"{'progressed' if outcome else 'did not progress'} (n={len(part):,})")
    ax.set_xlabel("Mean predicted risk over 30 passes")
    ax.set_ylabel("Uncertainty (SD over 30 passes)")
    ax.set_title("Uncertainty against predicted risk, by outcome")
    ax.legend(markerscale=3, fontsize=8)

    ax = axes[1]
    groups = [(label, part["mc_std"].to_numpy()) for label, part in df.groupby("lab_visit_group", observed=True)]
    ax.boxplot([g for _, g in groups], showfliers=False)
    ax.set_xticks(range(1, len(groups) + 1), [f"{label}\n(n={len(g):,})" for label, g in groups])
    ax.set_xlabel("Lab visits before the cutoff")
    ax.set_ylabel("Uncertainty (SD over 30 passes)")
    ax.set_title("Uncertainty by number of lab visits")
    fig.tight_layout()
    fig.savefig(rp.FIGURES_DIR / "uncertainty_mc_dropout.png", dpi=150)
    plt.close(fig)


def main() -> None:
    seed = rp.RANDOM_STATE
    torch.set_num_threads(os.cpu_count() or 1)
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    model = TimeDecayGRU(vocab_size=checkpoint["vocab_size"], n_static=2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    features = rp.load_feature_table()
    events = pd.read_parquet(SEQUENCES_PATH, columns=["patient_id", "date", "event_type", "token"])
    test_ids, y_test = sequence_splits(features)["test"]
    tensors = split_tensors(features, events, test_ids, checkpoint["token_to_id"], checkpoint["vocab_size"],
                            checkpoint["age_mean"], checkpoint["age_std"])

    deterministic = predict_proba(model, tensors)
    draws = mc_dropout_predict(model, tensors, passes=PASSES, seed=seed)
    df = pd.DataFrame({
        "patient_id": test_ids,
        "y_true": y_test,
        "deterministic": deterministic,
        "mc_mean": draws.mean(axis=0),
        "mc_std": draws.std(axis=0),
        "lab_visits": lab_visits_before_cutoff(events, test_ids),
    })
    df["lab_visit_group"] = pd.cut(df["lab_visits"], [-1, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"])
    df["abs_error"] = (df["y_true"] - df["mc_mean"]).abs()
    df.to_parquet(PREDICTIONS_PATH, index=False)
    plot_uncertainty(df)

    y, score, unc = df["y_true"].to_numpy(), df["mc_mean"].to_numpy(), df["mc_std"].to_numpy()
    # Uncertainty peaks near 50% risk, so also check it adds anything beyond |risk - 0.5|.
    margin = -np.abs(score - 0.5)
    summary = {
        "auroc_deterministic": roc_auc_score(y, df["deterministic"]),
        "auroc_mc_mean": roc_auc_score(y, score),
        "mean_std": unc.mean(),
        "mean_std_progressed": unc[y == 1].mean(),
        "mean_std_not_progressed": unc[y == 0].mean(),
        "spearman_std_vs_abs_error": spearmanr(unc, df["abs_error"]).statistic,
        "spearman_std_vs_lab_visits": spearmanr(unc, df["lab_visits"]).statistic,
        "spearman_std_vs_risk_margin": spearmanr(unc, margin).statistic,
        "auroc_most_certain_80pct": selective_auroc(y, score, unc, 0.8),
        "auroc_most_certain_50pct": selective_auroc(y, score, unc, 0.5),
        "auroc_margin_most_certain_50pct": selective_auroc(y, score, margin, 0.5),
        "spearman_std_vs_abs_error_within_risk_deciles": within_decile_spearman(df),
    }
    by_group = df.groupby("lab_visit_group", observed=True).agg(
        patients=("mc_std", "size"), mean_std=("mc_std", "mean"), median_std=("mc_std", "median"),
        mean_abs_error=("abs_error", "mean"), progression_rate=("y_true", "mean"),
    )
    by_group["auroc"] = [roc_auc_score(g["y_true"], g["mc_mean"]) for _, g in df.groupby("lab_visit_group", observed=True)]

    table = pd.DataFrame([{"target": TARGET, "model": "GRU MC dropout (30 passes)", **summary}])
    rp.save_metrics(table, rp.TABLES_DIR / "uncertainty_summary.csv")
    rp.save_metrics(by_group.reset_index(), rp.TABLES_DIR / "uncertainty_by_lab_visits.csv")
    write_results(table, task="8", seed=seed, metrics=list(summary),
                  notes="MC dropout, 30 passes, test split; uncertainty = SD of predicted risk")
    print(pd.Series(summary).round(4).to_string())
    print(by_group.round(4).to_string())


if __name__ == "__main__":
    main()
