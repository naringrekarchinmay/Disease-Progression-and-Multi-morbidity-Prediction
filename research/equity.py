"""Equity audit, baseline pass (ISEM 735 RQ4, proposal weeks 7-8).

RQ4 asks whether a scheduling policy moves monitoring away from subgroups that are
already less observed. Before any policy exists, this pass reports:

1. Historical monitoring by subgroup (is anyone already measured less?).
2. Predictor performance and calibration by subgroup, for the GRU and the calibrated
   XGBoost baseline, with bootstrap intervals.
3. Which subgroups an uncertainty-threshold rule would measure first, at several budgets.

The functions take plain prediction tables, so the same audit runs on the scheduling
policies later. Subgroups are sex, age band and insurance; the data has no race or
ethnicity field.

    python -m research.equity

Like research.gru_model, this module must never load XGBoost (see that module).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

import run_pipeline as rp

AGE_BINS = [0, 49, 64, 79, 200]
AGE_LABELS = ["<50", "50-64", "65-79", "80+"]
GROUP_COLUMNS = ["sex", "age_band", "insurance"]
BUDGETS = [0.10, 0.25, 0.50]
CUTOFF = pd.Timestamp("2021-12-31")
YEARS_OF_DATA = 7


def subgroups(patients: pd.DataFrame) -> pd.DataFrame:
    """One row per patient with the audit's subgroup columns."""
    return pd.DataFrame(
        {
            "patient_id": patients["patient_id"],
            "sex": patients["sex"],
            "age_band": pd.cut(patients["age"], AGE_BINS, labels=AGE_LABELS),
            "insurance": patients["insurance_type"],
        }
    )


def measured_share(df: pd.DataFrame, group_col: str, uncertainty_col: str, budgets: list[float]) -> pd.DataFrame:
    """Per subgroup, the share measured when the most uncertain `budget` fraction is measured.

    `ratio_to_budget` is the subgroup's selection rate over the budget: 1 is a fair
    share, below 1 means the subgroup is measured less than the population.
    """
    order = df[uncertainty_col].rank(method="first", ascending=False)
    rows = []
    for budget in budgets:
        selected = order <= round(budget * len(df))
        per_group = selected.groupby(df[group_col], observed=True).agg(["size", "mean"])
        for group, (size, rate) in per_group.iterrows():
            rows.append({"budget": budget, "group": group, "patients": int(size),
                         "selection_rate": rate, "ratio_to_budget": rate / budget})
    return pd.DataFrame(rows)


def performance_by_group(
    df: pd.DataFrame, group_col: str, y_col: str, score_col: str, n_boot: int = 200, seed: int = 42
) -> pd.DataFrame:
    """AUROC (with a percentile bootstrap 95% interval), Brier and calibration gap per subgroup.

    `calibration_gap` is observed rate minus mean predicted risk: positive means the model
    under-predicts that subgroup's risk.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for group, part in df.groupby(group_col, observed=True):
        y, score = part[y_col].to_numpy(), part[score_col].to_numpy()
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            if y[idx].min() != y[idx].max():
                boots.append(roc_auc_score(y[idx], score[idx]))
        rows.append({
            "group": group,
            "patients": len(y),
            "observed_rate": y.mean(),
            "mean_predicted": score.mean(),
            "calibration_gap": y.mean() - score.mean(),
            "brier": brier_score_loss(y, score),
            "auroc": roc_auc_score(y, score),
            "auroc_low": np.percentile(boots, 2.5),
            "auroc_high": np.percentile(boots, 97.5),
        })
    return pd.DataFrame(rows)


def historical_monitoring(groups: pd.DataFrame) -> pd.DataFrame:
    """Lab-visit rates per subgroup over 2018-2024, and share with no lab before the cutoff."""
    labs = pd.read_csv(rp.PROJECT_ROOT / "data" / "lab_results.csv", usecols=["patient_id", "test_date"],
                       parse_dates=["test_date"]).drop_duplicates()
    visits = labs.groupby("patient_id").size().reindex(groups["patient_id"], fill_value=0).to_numpy()
    before = labs[labs["test_date"] <= CUTOFF].groupby("patient_id").size()
    before = before.reindex(groups["patient_id"], fill_value=0).to_numpy()
    df = groups.assign(per_year=visits / YEARS_OF_DATA, none_before_cutoff=before == 0)
    rows = []
    for col in GROUP_COLUMNS:
        agg = df.groupby(col, observed=True).agg(
            patients=("per_year", "size"), lab_visits_per_year=("per_year", "mean"),
            share_no_lab_before_cutoff=("none_before_cutoff", "mean"))
        rows.append(agg.reset_index().rename(columns={col: "group"}).assign(attribute=col))
    table = pd.concat(rows, ignore_index=True)
    return table[["attribute", "group", "patients", "lab_visits_per_year", "share_no_lab_before_cutoff"]]


def plot_equity(monitoring: pd.DataFrame, performance: pd.DataFrame, shares: pd.DataFrame) -> None:
    labels = [f"{a}: {g}" for a, g in zip(monitoring["attribute"], monitoring["group"])]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)

    axes[0].barh(y, monitoring["lab_visits_per_year"], color="#4C72B0")
    axes[0].axvline(monitoring["lab_visits_per_year"].mean(), color="grey", linestyle=":")
    axes[0].set_xlabel("Lab visits per patient-year (2018-2024)")
    axes[0].set_title("Historical monitoring")

    for offset, (model, color) in zip([-0.18, 0.18], [("GRU", "#C44E52"), ("XGBoost + isotonic", "#4C72B0")]):
        part = performance[performance["model"] == model].set_index(["attribute", "group"])
        part = part.reindex(list(zip(monitoring["attribute"], monitoring["group"])))
        err = [part["auroc"] - part["auroc_low"], part["auroc_high"] - part["auroc"]]
        axes[1].errorbar(part["auroc"], y + offset, xerr=err, fmt="o", color=color, capsize=2, markersize=4, label=model)
    axes[1].set_xlabel("AUROC, test set (95% bootstrap interval)")
    axes[1].set_title("Predictor performance")
    axes[1].legend(fontsize=8, loc="lower left")

    for offset, (rule, color) in zip([-0.18, 0.18], [("dropout SD", "#DD8452"), ("closeness to 50% risk", "#55A868")]):
        part = shares[(shares["budget"] == 0.25) & (shares["rule"] == rule)].set_index(["attribute", "group"])
        part = part.reindex(list(zip(monitoring["attribute"], monitoring["group"])))
        axes[2].barh(y + offset, part["ratio_to_budget"], height=0.36, color=color, label=rule)
    axes[2].axvline(1, color="black", linewidth=1)
    axes[2].set_xlabel("Selection rate / budget at a 25% budget (1 = fair share)")
    axes[2].set_title("Who an uncertainty rule measures first")
    axes[2].legend(fontsize=8, loc="lower right")

    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(rp.FIGURES_DIR / "equity_baseline.png", dpi=150)
    plt.close(fig)


def main() -> None:
    from research.calibration import baseline_predictions_path, write_results
    from research.gru_model import TARGET
    from research.uncertainty import PREDICTIONS_PATH

    seed = rp.RANDOM_STATE
    patients = pd.read_csv(rp.PROJECT_ROOT / "data" / "patients.csv",
                           usecols=["patient_id", "age", "sex", "insurance_type"])
    groups = subgroups(patients)
    monitoring = historical_monitoring(groups)

    gru = pd.read_parquet(PREDICTIONS_PATH)[["patient_id", "y_true", "deterministic", "mc_std"]]
    xgb = pd.read_parquet(baseline_predictions_path(TARGET)).rename(columns={"y_score": "xgb"})
    test = gru.merge(xgb[["patient_id", "xgb"]], on="patient_id", validate="1:1").merge(groups, on="patient_id")
    test["closeness_to_50"] = -(test["deterministic"] - 0.5).abs()

    performance, shares = [], []
    for col in GROUP_COLUMNS:
        for model, score in [("GRU", "deterministic"), ("XGBoost + isotonic", "xgb")]:
            performance.append(performance_by_group(test, col, "y_true", score, n_boot=200, seed=seed)
                               .assign(attribute=col, model=model))
        for rule, unc in [("dropout SD", "mc_std"), ("closeness to 50% risk", "closeness_to_50")]:
            shares.append(measured_share(test, col, unc, BUDGETS).assign(attribute=col, rule=rule))
    performance, shares = pd.concat(performance, ignore_index=True), pd.concat(shares, ignore_index=True)

    rp.save_metrics(monitoring, rp.TABLES_DIR / "equity_monitoring.csv")
    rp.save_metrics(performance, rp.TABLES_DIR / "equity_performance.csv")
    rp.save_metrics(shares, rp.TABLES_DIR / "equity_measured_share.csv")
    plot_equity(monitoring, performance, shares)

    summary = []
    for col in GROUP_COLUMNS:
        mon = monitoring[monitoring["attribute"] == col]
        row = {"target": TARGET, "model": col,
               "monitoring_max_over_min": mon["lab_visits_per_year"].max() / mon["lab_visits_per_year"].min()}
        for model in ["GRU", "XGBoost + isotonic"]:
            perf = performance[(performance["attribute"] == col) & (performance["model"] == model)]
            key = "gru" if model == "GRU" else "xgb"
            row[f"{key}_auroc_range"] = perf["auroc"].max() - perf["auroc"].min()
            row[f"{key}_max_abs_calibration_gap"] = perf["calibration_gap"].abs().max()
        for rule, key in [("dropout SD", "sd"), ("closeness to 50% risk", "margin")]:
            sh = shares[(shares["attribute"] == col) & (shares["rule"] == rule) & (shares["budget"] == 0.25)]
            row[f"{key}_min_ratio_at_25pct"] = sh["ratio_to_budget"].min()
            row[f"{key}_max_ratio_at_25pct"] = sh["ratio_to_budget"].max()
        summary.append(row)
    summary = pd.DataFrame(summary)
    metrics = [c for c in summary.columns if c not in ("target", "model")]
    write_results(summary, task="12", seed=seed, metrics=metrics,
                  notes="baseline equity audit; model column holds the subgroup attribute; test split for predictors")

    print(monitoring.round(4).to_string(index=False))
    print(performance.round(4).to_string(index=False))
    print(summary.round(4).T.to_string())


if __name__ == "__main__":
    main()
