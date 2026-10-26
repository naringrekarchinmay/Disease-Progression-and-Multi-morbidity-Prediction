"""End-to-end smoke run of the simulator (ISEM 735 W6-3, results-log task 15).

For beta = 0 and beta = 1: simulate 100,000 patients, save the cohort, build the leak-free
features with the unchanged pipeline (src/features.py), refit the calibrated XGBoost with and
without lab features, and score masking schedules on the test split the same way
research/scheduling.py does on the Kaggle data. Expected: at beta 0 labs add nothing and every
schedule ties; at beta 1 labs carry clear signal.

    python -m research.simulator_smoke

This module loads XGBoost, so it must never import torch (see research/gru_model.py).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import run_pipeline as rp

TARGET = "multimorbidity_progression"
SIMULATED_DIR = rp.PROJECT_ROOT / "data" / "processed" / "simulated"
RANDOM_SEEDS = [1, 2, 3]
# Judgements for a smoke test on a 10,000-patient-scale test split (AUROC SE about 0.005).
FLAT_TOLERANCE = 0.005
SIGNAL_MARGIN = 0.01


def lab_columns(columns) -> list[str]:
    return [c for c in columns if c.startswith("lab_")]


def smoke_checks(table: pd.DataFrame) -> pd.DataFrame:
    """Judge the four W6-3 expectations from a table of (beta, kind, name, roc_auc)."""
    auc = table.set_index(["beta", "kind", "name"])["roc_auc"]

    def gain(beta: float) -> float:
        return auc[(beta, "model", "all features")] - auc[(beta, "model", "no lab features")]

    def schedules(beta: float) -> pd.Series:
        return auc.xs((beta, "schedule"), level=["beta", "kind"])

    s0, s1 = schedules(0.0), schedules(1.0)
    rows = [
        ("beta 0: lab features add nothing", gain(0.0), abs(gain(0.0)) <= FLAT_TOLERANCE),
        ("beta 0: masking schedules tie", s0.max() - s0.min(), s0.max() - s0.min() <= FLAT_TOLERANCE),
        ("beta 1: lab features add signal", gain(1.0), gain(1.0) >= SIGNAL_MARGIN),
        ("beta 1: keeping labs beats dropping them", s1["keep all"] - s1["keep none"],
         s1["keep all"] - s1["keep none"] >= SIGNAL_MARGIN),
    ]
    return pd.DataFrame(rows, columns=["check", "value", "passed"])


def save_cohort(tables: dict[str, pd.DataFrame], truth: pd.DataFrame, beta: float) -> None:
    out = SIMULATED_DIR / f"beta_{beta:g}"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(out / f"{name}.parquet", index=False)
    truth.to_parquet(out / "truth.parquet", index=False)


def run_beta(source: dict, params, beta: float, n_patients: int, seed: int) -> list[dict]:
    from research import simulator as sim
    from research.calibration import calibration_splits, fit_calibrated_xgboost, score_metrics
    from research.cost_model import SIMULATOR_MEASUREMENT_COST
    from research.scheduling import (INPUT_YEARS, SCORE_DATE, features_as_of, fixed_interval_keep,
                                     lab_opportunities, random_keep)
    from src.data_cleaning import clean_all
    from src.features import TARGET_COLUMNS, build_patient_features

    start = time.perf_counter()
    tables, truth = sim.simulate_cohort(source, params, n_patients, beta, seed)
    save_cohort(tables, truth, beta)
    data = clean_all(tables)
    features = build_patient_features(patients=data["patients"], diagnoses=data["diagnoses"],
                                      lab_results=data["lab_results"], medications=data["medications"],
                                      outcomes=data["outcomes"])
    print(f"beta={beta}: simulated and built {features.shape} features in {time.perf_counter() - start:.0f}s; "
          f"progression rate in at-risk cohort "
          f"{features.loc[features['baseline_condition_count'] < 2, TARGET].mean():.3f}")

    rows = []
    splits = calibration_splits(features, TARGET)
    X_test, y_test = splits["test"]
    models = {}
    for name, drop in [("all features", []), ("no lab features", lab_columns(X_test.columns))]:
        reduced = {k: (X.drop(columns=drop), y) for k, (X, y) in splits.items()}
        _, calibrated = fit_calibrated_xgboost(reduced, seed=rp.RANDOM_STATE)
        models[name] = calibrated
        risk = calibrated.predict_proba(reduced["test"][0])[:, 1]
        rows.append({"beta": beta, "kind": "model", "name": name, "features": X_test.shape[1] - len(drop),
                     **score_metrics(y_test, risk)})
        print(f"  {name}: AUROC {rows[-1]['roc_auc']:.4f} | {time.perf_counter() - start:.0f}s")

    # Masking schedules on the test split, scored with the all-features model (as research/scheduling.py).
    test_ids = features.loc[X_test.index, "patient_id"].tolist()
    ids = set(test_ids)
    subset = {name: t[t["patient_id"].isin(ids)] for name, t in data.items()}
    labs = subset["lab_results"]
    opps = lab_opportunities(labs, test_ids)
    columns = [c for c in features.columns if c not in TARGET_COLUMNS]
    as_of = pd.Series(SCORE_DATE, index=test_ids)

    def score(keep: pd.Series) -> dict:
        kept = opps.loc[keep.to_numpy(), ["patient_id", "date"]].rename(columns={"date": "test_date"})
        rows_ = features_as_of(subset, as_of, labs.merge(kept, on=["patient_id", "test_date"]), columns)
        risk = models["all features"].predict_proba(rows_.drop(columns="patient_id"))[:, 1]
        per_year = int(keep.sum()) / len(test_ids) / INPUT_YEARS
        return {"share_of_observed": float(keep.mean()), "spend_per_patient_year": per_year * SIMULATOR_MEASUREMENT_COST,
                **score_metrics(y_test.to_numpy(), risk)}

    twelve = fixed_interval_keep(opps, 365)
    schedules = {"keep all": pd.Series(True, index=opps.index), "keep none": pd.Series(False, index=opps.index),
                 "6-month": fixed_interval_keep(opps, 182), "12-month": twelve}
    for name, keep in schedules.items():
        rows.append({"beta": beta, "kind": "schedule", "name": name, **score(keep)})
    randoms = pd.DataFrame([score(random_keep(opps, int(twelve.sum()), s)) for s in RANDOM_SEEDS])
    rows.append({"beta": beta, "kind": "schedule", "name": "random (matched to 12-month)",
                 **randoms.mean().to_dict(), "roc_auc_sd": randoms["roc_auc"].std()})
    keep_all = next(r for r in rows if r["name"] == "keep all")["roc_auc"]
    direct = next(r for r in rows if r["name"] == "all features")["roc_auc"]
    assert abs(keep_all - direct) < 1e-9, (keep_all, direct)
    print(f"  schedules scored | {time.perf_counter() - start:.0f}s")
    return rows


def main() -> None:
    import argparse

    from research import simulator as sim
    from research.calibration import write_results

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patients", type=int, default=100000)
    parser.add_argument("--beta", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.perf_counter()
    source = sim.load_source()
    params = sim.fit_params(source)
    rows = []
    for beta in args.beta:
        rows += run_beta(source, params, beta, args.patients, args.seed)
    table = pd.DataFrame(rows)
    rp.save_metrics(table, rp.TABLES_DIR / "simulator_smoke.csv")
    show = ["beta", "kind", "name", "share_of_observed", "spend_per_patient_year", "roc_auc", "brier",
            "calibration_slope"]
    print(table.reindex(columns=show).round(4).to_string(index=False))
    if set(args.beta) >= {0.0, 1.0}:
        checks = smoke_checks(table)
        print(checks.round(4).to_string(index=False))
        rp.save_metrics(checks, rp.TABLES_DIR / "simulator_smoke_checks.csv")

    log = table.assign(target=TARGET, model="simulator beta=" + table["beta"].map("{:g}".format) + " "
                       + table["kind"] + ": " + table["name"])
    write_results(log, task="15", seed=args.seed, metrics=["roc_auc", "brier"],
                  notes=f"simulator smoke run, {args.patients:,} simulated patients; calibrated XGBoost refit per beta; "
                        "schedules masked on the test split")
    print(f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
