"""Evaluation spike (ISEM 735, Task 9): can this data evaluate measurement schedules?

1. Support: how many patients' lab histories already look like a 3-, 6- or
   12-month schedule (off-policy evaluation needs such patients).
2. Masking: how far the GRU's predicted risk moves when one lab visit is hidden.
3. The Delphi effort estimate lives in the Task 9 report, not in code.

    python -m research.evaluation_spike

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
from research.uncertainty import PREDICTIONS_PATH as MC_PREDICTIONS_PATH

TOLERANCE_DAYS = 30
MIN_GAPS = 2
SCHEDULES = {"3-month": 91, "6-month": 182, "12-month": 365}


def follows_schedule(
    lab_days: pd.DataFrame, interval_days: int, tolerance_days: int = TOLERANCE_DAYS, min_gaps: int = MIN_GAPS
) -> pd.Series:
    """True for patients with at least `min_gaps` gaps, every one within tolerance of the interval.

    `lab_days` has one row per patient and lab-visit date. Returns one bool per patient.
    """
    days = lab_days.drop_duplicates().sort_values(["patient_id", "date"])
    gaps = days.groupby("patient_id")["date"].diff().dt.days
    near = (gaps - interval_days).abs() <= tolerance_days
    per_patient = pd.DataFrame({"patient_id": days["patient_id"], "gap": gaps, "near": near}).dropna(subset=["gap"])
    summary = per_patient.groupby("patient_id").agg(n_gaps=("gap", "size"), all_near=("near", "all"))
    result = (summary["n_gaps"] >= min_gaps) & summary["all_near"]
    return result.reindex(days["patient_id"].unique(), fill_value=False)


def hide_lab_visit(events: pd.DataFrame, which: str = "last", seed: int = 42) -> pd.DataFrame:
    """Drop every lab event from one lab-visit day per patient; other events that day stay.

    `which` is "last" (the most recent lab visit before the cutoff) or "random".
    Patients without lab events are returned unchanged.
    """
    lab_days = events.loc[events["event_type"] == "lab", ["patient_id", "date"]].drop_duplicates()
    if which == "last":
        chosen = lab_days.groupby("patient_id")["date"].max()
    elif which == "random":
        rng = np.random.default_rng(seed)
        lab_days = lab_days.sort_values(["patient_id", "date"]).assign(key=lambda d: rng.random(len(d)))
        chosen = lab_days.loc[lab_days.groupby("patient_id")["key"].idxmin()].set_index("patient_id")["date"]
    else:
        raise ValueError(f"which must be 'last' or 'random', not {which!r}")
    hidden_date = events["patient_id"].map(chosen)
    drop = (events["event_type"] == "lab") & (events["date"] == hidden_date)
    return events[~drop]


def longest_near_run(lab_days: pd.DataFrame, interval_days: int, tolerance_days: int = TOLERANCE_DAYS) -> pd.Series:
    """Longest run of consecutive gaps within tolerance of the interval, per patient."""
    days = lab_days.drop_duplicates().sort_values(["patient_id", "date"])
    gaps = days.groupby("patient_id")["date"].diff().dt.days
    near = ((gaps - interval_days).abs() <= tolerance_days).astype(int)
    breaks = (near == 0).groupby(days["patient_id"]).cumsum()
    runs = near.groupby([days["patient_id"], breaks]).sum()
    return runs.groupby(level=0).max().reindex(days["patient_id"].unique(), fill_value=0)


def schedule_support() -> pd.DataFrame:
    """Patients whose 2018-2024 lab history resembles each schedule."""
    lab_days = pd.read_csv(rp.PROJECT_ROOT / "data" / "lab_results.csv", usecols=["patient_id", "test_date"],
                           parse_dates=["test_date"]).rename(columns={"test_date": "date"}).drop_duplicates()
    n_patients = lab_days["patient_id"].nunique()
    visits = lab_days.groupby("patient_id").size()
    rows = []
    for name, interval in SCHEDULES.items():
        follows = follows_schedule(lab_days, interval)
        runs = longest_near_run(lab_days, interval)
        full_schedule_visits = int(np.ceil(7 * 365 / interval)) + 1  # visits a full 2018-2024 schedule needs
        rows.append({
            "schedule": name,
            "interval_days": interval,
            "patients_with_3plus_visits": int((visits >= 3).sum()),
            "follows_schedule_all_gaps": int(follows.sum()),
            "share_of_all_patients": follows.sum() / n_patients,
            "run_of_2plus_near_gaps": int((runs >= 2).sum()),
            "run_of_3plus_near_gaps": int((runs >= 3).sum()),
            "visits_needed_for_full_schedule": full_schedule_visits,
            "patients_with_that_many_visits": int((visits >= full_schedule_visits).sum()),
        })
    return pd.DataFrame(rows)


def masking_effect() -> tuple[pd.DataFrame, pd.DataFrame]:
    """GRU risk change on the test split when one pre-cutoff lab visit is hidden."""
    torch.set_num_threads(os.cpu_count() or 1)
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    model = TimeDecayGRU(vocab_size=checkpoint["vocab_size"], n_static=2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    features = rp.load_feature_table()
    events = pd.read_parquet(SEQUENCES_PATH, columns=["patient_id", "date", "event_type", "token"])
    test_ids, y_test = sequence_splits(features)["test"]
    events = events[events["patient_id"].isin(set(test_ids))]

    def score(ev: pd.DataFrame) -> np.ndarray:
        tensors = split_tensors(features, ev, test_ids, checkpoint["token_to_id"], checkpoint["vocab_size"],
                                checkpoint["age_mean"], checkpoint["age_std"])
        return predict_proba(model, tensors)

    original = score(events)
    has_labs = pd.Series(test_ids).isin(set(events.loc[events["event_type"] == "lab", "patient_id"])).to_numpy()
    mc_sd = pd.read_parquet(MC_PREDICTIONS_PATH).set_index("patient_id").loc[test_ids, "mc_std"].to_numpy()

    per_patient = pd.DataFrame({"patient_id": test_ids, "y_true": y_test, "has_labs": has_labs,
                                "original": original, "mc_sd": mc_sd})
    rows = []
    for which in ["last", "random"]:
        masked = score(hide_lab_visit(events, which=which, seed=rp.RANDOM_STATE))
        per_patient[f"delta_{which}"] = masked - original
        d = per_patient.loc[has_labs, f"delta_{which}"]
        crossed = ((original >= 0.5) != (masked >= 0.5))[has_labs]
        rows.append({
            "hidden_visit": which,
            "patients_with_labs": int(has_labs.sum()),
            "median_abs_change": d.abs().median(),
            "p90_abs_change": d.abs().quantile(0.9),
            "max_abs_change": d.abs().max(),
            "mean_signed_change": d.mean(),
            "share_abs_change_over_0.05": (d.abs() > 0.05).mean(),
            "share_abs_change_over_0.10": (d.abs() > 0.10).mean(),
            "share_crossing_0.5": crossed.mean(),
            "median_change_over_mc_sd": (d.abs() / per_patient.loc[has_labs, "mc_sd"]).median(),
            "auroc_original": roc_auc_score(y_test, original),
            "auroc_masked": roc_auc_score(y_test, masked),
        })
    return pd.DataFrame(rows), per_patient


def plot_masking(per_patient: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    d = per_patient[per_patient["has_labs"]]
    bins = np.linspace(-0.2, 0.2, 81)
    for which, color in [("last", "#C44E52"), ("random", "#4C72B0")]:
        ax.hist(d[f"delta_{which}"].clip(-0.2, 0.2), bins=bins, alpha=0.55, color=color,
                label=f"hide {which} lab visit (median |change| {d[f'delta_{which}'].abs().median():.3f})")
    ax.axvline(-d["mc_sd"].mean(), color="grey", linestyle=":", linewidth=1)
    ax.axvline(d["mc_sd"].mean(), color="grey", linestyle=":", linewidth=1, label=f"± mean MC-dropout SD ({d['mc_sd'].mean():.3f})")
    ax.set_xlabel("Change in predicted risk when one lab visit is hidden (clipped to ±0.2)")
    ax.set_ylabel("Test patients")
    ax.set_title(f"How much one lab visit moves the GRU's risk ({len(d):,} test patients with labs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rp.FIGURES_DIR / "spike_masking_effect.png", dpi=150)
    plt.close(fig)


def main() -> None:
    support = schedule_support()
    rp.save_metrics(support, rp.TABLES_DIR / "spike_schedule_support.csv")
    print(support.round(4).to_string(index=False))

    masking, per_patient = masking_effect()
    rp.save_metrics(masking, rp.TABLES_DIR / "spike_masking_effect.csv")
    plot_masking(per_patient)
    print(masking.round(4).T.to_string())

    support_log = support.assign(target="lab schedule support", model=support["schedule"])
    masking_log = masking.assign(target=TARGET, model="GRU, hide " + masking["hidden_visit"] + " lab visit")
    write_results(
        pd.concat([
            support_log[["target", "model", "follows_schedule_all_gaps", "run_of_2plus_near_gaps"]],
            masking_log[["target", "model", "median_abs_change", "p90_abs_change", "share_crossing_0.5", "auroc_masked"]],
        ]),
        task="9", seed=rp.RANDOM_STATE,
        metrics=["follows_schedule_all_gaps", "run_of_2plus_near_gaps", "median_abs_change", "p90_abs_change",
                 "share_crossing_0.5", "auroc_masked"],
        notes="schedule support: all 100k patients, 2018-2024, +/-30 days; masking: GRU test split",
    )


if __name__ == "__main__":
    main()
