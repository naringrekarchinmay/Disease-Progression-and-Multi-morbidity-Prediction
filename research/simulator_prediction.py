"""Beta sweep, part 2: prediction under each schedule (ISEM 735 W7-2, results-log task 17).

A denser schedule diagnoses people earlier, which changes the diagnosis-based progression label and
the at-risk cohort. So the label here comes from the simulator's truth, the same for every schedule:
a patient is at risk if fewer than two conditions have truly begun by 2021-12-31 (true onset for the
four lab-linked conditions, the copied diagnosis date for the other 11), and progresses if two or more
have begun by 2024-12-31.

Features are what each schedule would have observed by the cutoff: its lab visits up to 2021-12-31
(full panel, common random numbers as in research/simulator_sweep.py), the diagnoses those visits and
clinical presentation produced, and medications from diagnosis. The unchanged feature pipeline
(src/features.py) builds them, in patient chunks to bound memory. Each schedule gets its own
calibrated XGBoost (a model deployed under a schedule is trained on data collected under it), on one
patient split shared by every schedule, so AUROC differences are paired.

    python -m research.simulator_prediction

This module loads XGBoost, so it must never import torch (see research/gru_model.py).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from research import simulator as sim
from research import simulator_sweep as sweep

CUTOFF = pd.Timestamp("2021-12-31")
CUTOFF_MONTH = 47  # December 2021
TEST_SIZE = 0.25
VALIDATION_SIZE = 0.2
N_BOOT = 200
CHUNK = 20000


def true_labels(cohort: sweep.Cohort, source: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """True condition counts at the cutoff and at the end of 2024, and the progression label."""
    from src.data_cleaning import make_condition_events

    events = make_condition_events(source["diagnoses"])
    events = events[~events["condition"].isin(sim.LAB_CONDITIONS)]
    first = events.groupby(["patient_id", "condition"])["visit_date"].min().reset_index()
    other = cohort.mapping.merge(first.rename(columns={"patient_id": "source_patient_id"}), on="source_patient_id")
    ids = cohort.mapping["patient_id"]
    other_base = other[other["visit_date"] <= CUTOFF].groupby("patient_id").size().reindex(ids, fill_value=0)
    other_end = other[other["visit_date"] <= sim.END_DATE].groupby("patient_id").size().reindex(ids, fill_value=0)
    onset = cohort.truth
    linked_base = onset[onset["onset_month"] <= CUTOFF_MONTH].groupby("patient_id").size().reindex(ids, fill_value=0)
    linked_end = onset[onset["onset_month"] <= sim.N_MONTHS - 1].groupby("patient_id").size().reindex(ids, fill_value=0)
    out = pd.DataFrame({"patient_id": ids.to_numpy(),
                        "true_baseline_count": (other_base + linked_base).to_numpy(),
                        "true_future_count": (other_end + linked_end).to_numpy()})
    out["at_risk"] = out["true_baseline_count"] < 2
    out["label"] = (out["at_risk"] & (out["true_future_count"] >= 2)).astype(int)
    return out


def schedule_features(source: dict[str, pd.DataFrame], params: sim.SimulatorParams, cohort: sweep.Cohort,
                      visits: pd.DataFrame, beta: float, seed: int, chunk: int = CHUNK) -> pd.DataFrame:
    """Feature table (src/features.py) for the schedule's pre-cutoff visits, built in patient chunks."""
    from src.data_cleaning import clean_all
    from src.features import build_patient_features

    visits = visits[visits["test_date"] <= CUTOFF]
    ids = cohort.mapping["patient_id"].to_numpy()
    parts = []
    for start in range(0, len(ids), chunk):
        part_ids = set(ids[start:start + chunk])
        mapping = cohort.mapping[cohort.mapping["patient_id"].isin(part_ids)]
        truth = cohort.truth[cohort.truth["patient_id"].isin(part_ids)]
        tables, _ = sim.build_tables(source, params, mapping, truth, visits[visits["patient_id"].isin(part_ids)],
                                     beta, np.random.default_rng(seed + start), crn_seed=seed)
        data = clean_all(tables)
        parts.append(build_patient_features(patients=data["patients"], diagnoses=data["diagnoses"],
                                            lab_results=data["lab_results"], medications=data["medications"],
                                            outcomes=data["outcomes"]))
    features = pd.concat(parts, ignore_index=True)
    flags = [c for c in features.columns if c.startswith(("baseline_dx_", "med_indication_"))]
    features[flags] = features[flags].fillna(0).astype(int)
    if "lab_test_count" in features.columns:
        features[["lab_test_count", "lab_unique_test_count"]] = features[["lab_test_count", "lab_unique_test_count"]].fillna(0)
    return features.sort_values("patient_id").reset_index(drop=True)


def split_ids(labels: pd.DataFrame, seed: int = 42) -> dict[str, list[str]]:
    """Fit / validation / test patient ids within the true at-risk cohort, stratified on the label."""
    from sklearn.model_selection import train_test_split

    cohort = labels[labels["at_risk"]].sort_values("patient_id")
    train, test = train_test_split(cohort, test_size=TEST_SIZE, random_state=seed, stratify=cohort["label"])
    fit, val = train_test_split(train, test_size=VALIDATION_SIZE, random_state=seed, stratify=train["label"])
    return {"fit": fit["patient_id"].tolist(), "validation": val["patient_id"].tolist(),
            "test": test["patient_id"].tolist()}


def fit_and_predict(features: pd.DataFrame, labels: pd.DataFrame, split: dict[str, list[str]],
                    seed: int = 42) -> pd.Series:
    """Calibrated XGBoost fitted on this schedule's features; risk on the test ids."""
    from research.calibration import fit_calibrated_xgboost
    from src.features import TARGET_COLUMNS

    table = features.drop(columns=[c for c in TARGET_COLUMNS if c in features.columns]).set_index("patient_id")
    y = labels.set_index("patient_id")["label"]
    splits = {part: (table.loc[ids], y.loc[ids]) for part, ids in split.items()}
    _, calibrated = fit_calibrated_xgboost(splits, seed=seed)
    return pd.Series(calibrated.predict_proba(splits["test"][0])[:, 1], index=split["test"])


def paired_auroc_se(y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = 42) -> float:
    """Bootstrap standard error of AUROC(a) - AUROC(b) on the same test patients."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return float(np.std(diffs, ddof=1))


def schedules_for(cohort: sweep.Cohort, params: sim.SimulatorParams, beta: float, seed: int) -> dict:
    ids = pd.Series(cohort.ages.index)
    out = {"no labs": cohort.observed.iloc[:0], "observed care": cohort.observed}
    for name, interval in sweep.FIXED_INTERVALS.items():
        out[name] = sweep.visits_from_months(sweep.fixed_schedule(ids, interval, seed))
    out["random (matched to 12-month)"] = sweep.visits_from_months(
        sweep.random_schedule(ids, len(out["12-month"]), sweep.RANDOM_SEEDS[0]))
    out["guideline"] = sweep.run_guideline(cohort, params, beta, seed)[2]
    return out


def main() -> None:
    import argparse

    import run_pipeline as rp
    from research.calibration import score_metrics, write_results
    from research.cost_model import SIMULATOR_MEASUREMENT_COST

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patients", type=int, default=100000)
    parser.add_argument("--beta", type=float, nargs="+", default=sweep.BETAS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.perf_counter()
    source = sim.load_source()
    params = sim.fit_params(source)
    cohort = sweep.cohort_truth(source, params, args.patients, args.seed)
    labels = true_labels(cohort, source)
    split = split_ids(labels, args.seed)
    y_test = labels.set_index("patient_id").loc[split["test"], "label"].to_numpy()
    print(f"true at-risk {labels['at_risk'].sum():,}, progression {labels.loc[labels['at_risk'], 'label'].mean():.3f}, "
          f"test {len(split['test']):,} | {time.perf_counter() - start:.0f}s")

    rows = []
    for beta in args.beta:
        risks = {}
        for name, visits in schedules_for(cohort, params, beta, args.seed).items():
            features = schedule_features(source, params, cohort, visits, beta, args.seed)
            risks[name] = fit_and_predict(features, labels, split, args.seed).to_numpy()
            pre = visits[visits["test_date"] <= CUTOFF]
            per_year = len(pre) / args.patients / 4
            rows.append({"beta": beta, "schedule": name, "pre_cutoff_measurements_per_patient_year": per_year,
                         "spend_per_patient_year": per_year * SIMULATOR_MEASUREMENT_COST,
                         **score_metrics(y_test, risks[name])})
            print(f"  beta={beta} {name}: AUROC {rows[-1]['roc_auc']:.4f} Brier {rows[-1]['brier']:.4f} | "
                  f"{time.perf_counter() - start:.0f}s")
        for row in rows:
            if row["beta"] == beta:
                for ref in ["observed care", "no labs"]:
                    row[f"auroc_diff_vs_{ref}"] = row["roc_auc"] - next(
                        r["roc_auc"] for r in rows if r["beta"] == beta and r["schedule"] == ref)
                    row[f"auroc_diff_se_vs_{ref}"] = (0.0 if row["schedule"] == ref else
                                                      paired_auroc_se(y_test, risks[row["schedule"]], risks[ref]))
    table = pd.DataFrame(rows)
    rp.save_metrics(table, rp.TABLES_DIR / "simulator_sweep_prediction.csv")
    show = ["beta", "schedule", "spend_per_patient_year", "roc_auc", "brier", "calibration_slope",
            "auroc_diff_vs_observed care", "auroc_diff_se_vs_observed care", "auroc_diff_vs_no labs"]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(table[show].round(4).to_string(index=False))
    log = table.assign(target="true-onset multimorbidity progression",
                       model="beta=" + table["beta"].map("{:g}".format) + " " + table["schedule"])
    write_results(log, task="17", seed=args.seed, metrics=["spend_per_patient_year", "roc_auc", "brier"],
                  notes=f"simulator prediction sweep, {args.patients:,} patients; labels from true onsets; "
                        "calibrated XGBoost refit per schedule on a shared split; pre-cutoff measurements only")
    print(f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
