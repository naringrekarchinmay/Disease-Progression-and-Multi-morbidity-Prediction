"""Confidence intervals and seed spread for Tables 1 and 2 (ISEM 735, task 24).

Table 1 (RQ1): calibrated XGBoost, GRU and Transformer on the multimorbidity test split.
Reports patient-level bootstrap CIs for AUROC and Brier, paired AUROC differences with a
bootstrap CI and a DeLong test, and the spread over five training seeds per model (XGBoost
refitted here; the sequence models come from research/seed_runs.py).

Table 2 (RQ2 under masking): per-patient risks saved by research/scheduling.run_frontier for
each lab version (task 23). Reports bootstrap CIs for each schedule's AUROC and paired
differences: labs vs no labs, 12-month vs observed, and each threshold rule vs its
budget-matched random schedules (averaged over the five random seeds inside every resample).

    python -m research.model_ci

This module loads XGBoost, so it must never import torch.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import run_pipeline as rp
from research.calibration import calibration_splits, fit_calibrated_xgboost, write_results
from research.data import load_features
from research.scheduling import RANDOM_SEEDS, TARGET, TAUS, baseline_path, risks_path
from research.seed_paths import OUT_DIR, SEEDS, predictions_path
from research.stats_ci import bootstrap_indices, bootstrap_metrics, delong, paired_difference, summarize
from sklearn.metrics import brier_score_loss, roc_auc_score

LAB_VERSIONS = {"original": "all tests (original)", "panel": "universal panel",
                "panel_w": "universal panel, constrained to W"}
MODELS = ["XGBoost", "GRU", "Transformer"]


def xgboost_seed_predictions() -> None:
    features = load_features()
    splits = calibration_splits(features, TARGET)
    X_test, y_test = splits["test"]
    for seed in SEEDS:
        path = predictions_path("XGBoost", seed)
        if path.exists():
            continue
        _, calibrated = fit_calibrated_xgboost(splits, seed=seed)
        pd.DataFrame({"patient_id": features.loc[X_test.index, "patient_id"], "y_true": y_test,
                      "y_score": calibrated.predict_proba(X_test)[:, 1]}).to_parquet(path, index=False)


def load_predictions(seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    frames = {m: pd.read_parquet(predictions_path(m, seed)).set_index("patient_id") for m in MODELS}
    ids = frames["XGBoost"].index
    y = frames["XGBoost"]["y_true"].to_numpy()
    for m, f in frames.items():
        assert (f.loc[ids, "y_true"].to_numpy() == y).all(), f"{m} test split differs"
    return y, {m: frames[m].loc[ids, "y_score"].to_numpy() for m in MODELS}


def table1(idx: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    y, scores = load_predictions(42)
    rows = []
    for metric in ["roc_auc", "brier"]:
        boot = bootstrap_metrics(y, scores, idx, metric)
        fn = roc_auc_score if metric == "roc_auc" else brier_score_loss
        for m in MODELS:
            rows.append({"comparison": m, "metric": metric, **summarize(fn(y, scores[m]), boot[m].to_numpy())})
    for a, b in [("XGBoost", "GRU"), ("XGBoost", "Transformer"), ("GRU", "Transformer")]:
        d = delong(y, scores[a], scores[b])
        rows.append({"comparison": f"{a} - {b}", "metric": "roc_auc",
                     **paired_difference(y, scores[a], scores[b], idx),
                     "delong_se": d["se"], "delong_p": d["p_value"]})
    point = pd.DataFrame(rows)

    seed_rows = []
    for seed in SEEDS:
        y, s = load_predictions(seed)
        row = {"seed": seed, **{f"{m} roc_auc": roc_auc_score(y, s[m]) for m in MODELS},
               **{f"{m} brier": brier_score_loss(y, s[m]) for m in MODELS}}
        for a, b in [("XGBoost", "GRU"), ("XGBoost", "Transformer")]:
            d = delong(y, s[a], s[b])
            row[f"{a} - {b} diff"], row[f"{a} - {b} delong_p"] = d["diff"], d["p_value"]
        seed_rows.append(row)
    seeds = pd.DataFrame(seed_rows)
    return point, seeds


def table2(idx: np.ndarray) -> pd.DataFrame:
    rows = []
    for version, label in LAB_VERSIONS.items():
        risks = pd.read_parquet(risks_path(version))
        y = risks["y_true"].to_numpy()
        col = {c: risks[c].to_numpy() for c in risks.columns if c not in ("patient_id", "y_true")}
        assert abs(roc_auc_score(y, col["keep all (observed)"])
                   - roc_auc_score(*pd.read_parquet(baseline_path(version))[["y_true", "y_score"]].T.to_numpy())) < 1e-9
        tests = [("keep all - keep none", col["keep all (observed)"], col["keep none"]),
                 ("12-month - keep all", col["fixed interval 12-month"], col["keep all (observed)"]),
                 ("6-month - keep all", col["fixed interval 6-month"], col["keep all (observed)"])]
        for tau in TAUS:
            randoms = [col[f"random matched to tau={tau} seed={s}"] for s in RANDOM_SEEDS]
            tests.append((f"threshold tau={tau} - matched random", col[f"threshold tau={tau}"], randoms))
        for name, a, b in tests:
            rows.append({"labs": label, "comparison": name, **paired_difference(y, a, b, idx)})
        print(f"  table 2, {label} done", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [predictions_path(m, s) for m in ["GRU", "Transformer"] for s in SEEDS
               if not predictions_path(m, s).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} sequence-model seed runs missing; run python -m research.seed_runs")
    xgboost_seed_predictions()
    reference = pd.read_parquet(baseline_path("original"))
    seed42 = pd.read_parquet(predictions_path("XGBoost", 42))
    assert np.allclose(seed42["y_score"].to_numpy(), reference["y_score"].to_numpy()), "XGBoost seed 42 differs"
    print(f"XGBoost seeds ready | {time.perf_counter() - start:.0f}s", flush=True)

    n_test = len(reference)
    idx = bootstrap_indices(n_test)
    point, seeds = table1(idx)
    print(point.round(4).to_string(index=False))
    summary = seeds.drop(columns="seed").agg(["mean", "std", "min", "max"]).T
    print(summary.round(4).to_string())
    risks_n = len(pd.read_parquet(risks_path("original")))
    t2 = table2(idx if risks_n == n_test else bootstrap_indices(risks_n))
    print(t2[["labs", "comparison", "estimate", "ci_low", "ci_high", "share_above_zero"]].round(4).to_string(index=False))

    rp.save_metrics(point, rp.TABLES_DIR / "table1_bootstrap_ci.csv")
    rp.save_metrics(seeds, rp.TABLES_DIR / "table1_seeds.csv")
    rp.save_metrics(t2, rp.TABLES_DIR / "table2_bootstrap_ci.csv")
    log = pd.concat([
        point.assign(target=TARGET, model="table 1 " + point["comparison"] + " " + point["metric"]),
        t2.assign(target=TARGET, model="table 2 " + t2["labs"] + ": " + t2["comparison"]),
    ], ignore_index=True)
    write_results(log, task="24", seed=rp.RANDOM_STATE, metrics=["estimate", "ci_low", "ci_high"],
                  notes="patient-level paired bootstrap, 2,000 resamples, 95% percentile CI; seeds 42,1,2,3,4 in "
                        "outputs/tables/table1_seeds.csv")
    print(f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
