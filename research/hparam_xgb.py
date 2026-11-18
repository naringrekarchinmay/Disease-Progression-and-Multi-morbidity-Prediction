"""XGBoost hyperparameter search, five-seed refits and the tuned Table 1 (ISEM 735, task 29).

Protocol: research/hparam_notes.md section 1. 50 Optuna TPE trials scored by raw validation log
loss, trial 0 = the default settings, XGBoost unweighted (scale_pos_weight 1) so its raw
probabilities are scored on the same footing as the torch models. The best configuration is
refitted with seeds 42, 1, 2, 3, 4 and isotonic-calibrated on validation, as Table 1 does.
The tuned table then compares all three tuned models (GRU and Transformer predictions come from
research/hparam_torch.py) and each model's tuned minus default AUROC.

    python -m research.hparam_torch   # first, in its own process
    python -m research.hparam_xgb

This module loads XGBoost, so it must never import torch.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import run_pipeline as rp
from research.calibration import calibration_splits, write_results
from research.data import load_features
from research.hparam_paths import (N_TRIALS, OUT_DIR, TRAIN_SEED, load_best, make_study, predictions_path,
                                   save_study, suggest)
from research.model_ci import MODELS, load_predictions
from research.scheduling import TARGET
from research.seed_paths import SEEDS
from research.seed_paths import predictions_path as default_predictions_path
from research.stats_ci import bootstrap_indices, bootstrap_metrics, delong, paired_difference, summarize
from src.modeling import build_preprocessor


def make_xgboost(X: pd.DataFrame, params: dict, seed: int) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(X)),
                     ("model", XGBClassifier(**params, eval_metric="logloss", scale_pos_weight=1.0,
                                             random_state=seed, n_jobs=-1))])


def search(splits: dict, n_trials: int = N_TRIALS) -> dict:
    X_fit, y_fit = splits["fit"]
    X_val, y_val = splits["validation"]

    def objective(trial: optuna.Trial) -> float:
        model = make_xgboost(X_fit, suggest(trial, "XGBoost"), TRAIN_SEED).fit(X_fit, y_fit)
        return float(log_loss(y_val, model.predict_proba(X_val)[:, 1]))

    study = make_study("XGBoost")
    study.optimize(objective, n_trials=n_trials)
    return save_study(study, "XGBoost")


def refit(features: pd.DataFrame, splits: dict) -> None:
    params = load_best("XGBoost")
    X_fit, y_fit = splits["fit"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]
    for seed in SEEDS:
        base = make_xgboost(X_fit, params, seed).fit(X_fit, y_fit)
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(X_val, y_val)
        pd.DataFrame({"patient_id": features.loc[X_test.index, "patient_id"], "y_true": y_test,
                      "y_score": calibrated.predict_proba(X_test)[:, 1]}).to_parquet(
            predictions_path("XGBoost", seed), index=False)


def tuned_table(idx: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    y, tuned = load_predictions(42, predictions_path)
    y_default, default = load_predictions(42, default_predictions_path)
    assert (y == y_default).all(), "tuned and default test splits differ"
    rows = []
    for metric in ["roc_auc", "brier"]:
        fn = roc_auc_score if metric == "roc_auc" else brier_score_loss
        both = {**{f"{m} tuned": tuned[m] for m in MODELS}, **{f"{m} default": default[m] for m in MODELS}}
        boot = bootstrap_metrics(y, both, idx, metric)
        for name, s in both.items():
            rows.append({"comparison": name, "metric": metric, **summarize(fn(y, s), boot[name].to_numpy())})
    for a, b in [("XGBoost", "GRU"), ("XGBoost", "Transformer"), ("GRU", "Transformer")]:
        d = delong(y, tuned[a], tuned[b])
        rows.append({"comparison": f"{a} - {b} (tuned)", "metric": "roc_auc",
                     **paired_difference(y, tuned[a], tuned[b], idx), "delong_se": d["se"], "delong_p": d["p_value"]})
    for m in MODELS:
        d = delong(y, tuned[m], default[m])
        rows.append({"comparison": f"{m} tuned - default", "metric": "roc_auc",
                     **paired_difference(y, tuned[m], default[m], idx), "delong_se": d["se"], "delong_p": d["p_value"]})
        rows.append({"comparison": f"{m} tuned - default", "metric": "brier",
                     **paired_difference(y, tuned[m], default[m], idx, metric="brier")})
    point = pd.DataFrame(rows)

    rival = max(["GRU", "Transformer"], key=lambda m: roc_auc_score(y, tuned[m]))
    seed_rows = []
    for seed in SEEDS:
        ys, s = load_predictions(seed, predictions_path)
        row = {"seed": seed, **{f"{m} roc_auc": roc_auc_score(ys, s[m]) for m in MODELS},
               **{f"{m} brier": brier_score_loss(ys, s[m]) for m in MODELS}}
        for b in ["GRU", "Transformer"]:
            d = delong(ys, s["XGBoost"], s[b])
            row[f"XGBoost - {b} diff"], row[f"XGBoost - {b} delong_p"] = d["diff"], d["p_value"]
        seed_rows.append(row)
    seeds = pd.DataFrame(seed_rows)

    d = point[(point["comparison"] == f"XGBoost - {rival} (tuned)")].iloc[0]
    verdict = ("holds: XGBoost ahead" if d["ci_low"] > 0 else
               "sequence model ahead" if d["ci_high"] < 0 else "no difference")
    decision = {"rival": rival, "D": d["estimate"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                "seeds_D_positive": int((seeds[f"XGBoost - {rival} diff"] > 0).sum()), "verdict": verdict}
    return point, seeds, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--skip-search", action="store_true", help="Reuse the saved best configuration.")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    features = load_features()
    splits = calibration_splits(features, TARGET)
    if not args.skip_search:
        best = search(splits, args.trials)
        print(f"XGBoost: best trial {best['trial']} val log loss {best['val_log_loss']:.4f} "
              f"(default {best['default_val_log_loss']:.4f}) {best['params']} | {time.perf_counter() - start:.0f}s")
    refit(features, splits)
    missing = [predictions_path(m, s) for m in MODELS for s in SEEDS if not predictions_path(m, s).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} tuned runs missing; run python -m research.hparam_torch")

    point, seeds, decision = tuned_table(bootstrap_indices(len(pd.read_parquet(predictions_path("XGBoost", 42)))))
    print(point.round(4).to_string(index=False))
    print(seeds.drop(columns="seed").agg(["mean", "std", "min", "max"]).T.round(4).to_string())
    print(f"RQ1 decision: {decision}")
    rp.save_metrics(point, rp.TABLES_DIR / "table1_tuned_bootstrap_ci.csv")
    rp.save_metrics(seeds, rp.TABLES_DIR / "table1_tuned_seeds.csv")
    write_results(point.assign(target=TARGET, model="table 1 tuned " + point["comparison"] + " " + point["metric"]),
                  task="29", seed=rp.RANDOM_STATE, metrics=["estimate", "ci_low", "ci_high"],
                  notes=f"Optuna TPE, {args.trials} trials per model, validation log loss; refit seeds 42,1,2,3,4; "
                        f"RQ1 vs {decision['rival']}: {decision['verdict']}")
    print(f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
