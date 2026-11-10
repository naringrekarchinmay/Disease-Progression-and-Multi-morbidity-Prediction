"""Table 3 for the deployed model: uncertainty signals from XGBoost (ISEM 735, task 27).

Part 3 feedback, major issue 3: scheduling uses the calibrated XGBoost, but Table 3 (which age groups an
uncertainty rule measures first, research/equity.py) used the GRU's signals. It also asked that the two
signals be named for what they are:

- **model disagreement** (epistemic uncertainty): the standard deviation of predicted risk across
  10 XGBoost models, each fitted to a patient bootstrap of the fit split, as in the simulator's spread
  rule (research/simulator_policy.train_risk_models). For the GRU this was the MC-dropout SD.
- **risk ambiguity**: closeness of the calibrated risk to 50%. It measures how near a patient sits to
  the decision boundary, not how unsure the model is about its estimate.

Both are scored on the multimorbidity test split by age band, sex and insurance at budgets of 10%, 25%
and 50%, with patient-bootstrap intervals for the ratio to a fair share.

    python -m research.equity_xgb

This module loads XGBoost, so it must never import torch.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import run_pipeline as rp
from research.calibration import baseline_predictions_path, calibration_splits, write_results
from research.data import load_features
from research.equity import BUDGETS, GROUP_COLUMNS, measured_share, subgroups
from src.modeling import make_advanced_models

TARGET = "multimorbidity_progression"
N_BAGS = 10
N_BOOT = 500
SIGNALS = {"model disagreement (ensemble SD)": "spread", "risk ambiguity (closeness to 50%)": "closeness"}


def bagged_spread(splits, seed: int = rp.RANDOM_STATE, n_bags: int = N_BAGS) -> np.ndarray:
    """SD across XGBoost models fitted to patient bootstraps of the fit split, on the test split."""
    X_fit, y_fit = splits["fit"]
    X_test, _ = splits["test"]
    rng = np.random.default_rng(seed)
    preds = []
    for b in range(n_bags):
        rows = rng.integers(0, len(X_fit), len(X_fit))
        Xb, yb = X_fit.iloc[rows], y_fit.iloc[rows]
        model = make_advanced_models(Xb, random_state=seed + b, pos_weight=rp._pos_weight(yb))["xgboost"]
        model.fit(Xb, yb)
        preds.append(model.predict_proba(X_test)[:, 1])
    return np.std(preds, axis=0)


def ratio_intervals(test: pd.DataFrame, col: str, signal: str, n_boot: int = N_BOOT,
                    seed: int = rp.RANDOM_STATE) -> pd.DataFrame:
    """Percentile intervals for each group's ratio to a fair share, resampling patients."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        part = test.iloc[rng.integers(0, len(test), len(test))].reset_index(drop=True)
        draws.append(measured_share(part, col, signal, BUDGETS)[["budget", "group", "ratio_to_budget"]])
    d = pd.concat(draws)
    return (d.groupby(["budget", "group"])["ratio_to_budget"].quantile([0.025, 0.975]).unstack()
            .rename(columns={0.025: "ratio_ci_low", 0.975: "ratio_ci_high"}).reset_index())


def main() -> None:
    start = time.perf_counter()
    features = load_features()
    splits = calibration_splits(features, TARGET)
    X_test, y_test = splits["test"]
    xgb = pd.read_parquet(baseline_predictions_path(TARGET))
    assert (xgb["patient_id"].to_numpy() == features.loc[X_test.index, "patient_id"].to_numpy()).all()
    patients = pd.read_csv(rp.PROJECT_ROOT / "data" / "patients.csv",
                           usecols=["patient_id", "age", "sex", "insurance_type"])
    test = xgb.merge(subgroups(patients), on="patient_id", validate="1:1")
    test["spread"] = bagged_spread(splits)
    test["closeness"] = -(test["y_score"] - 0.5).abs()
    print(f"bagged spread ready | {time.perf_counter() - start:.0f}s", flush=True)
    rho = pd.Series(test["spread"]).corr(test["y_score"], method="spearman")
    print(f"Spearman(spread, risk) = {rho:.3f}")

    rows = []
    for col in GROUP_COLUMNS:
        for name, signal in SIGNALS.items():
            share = measured_share(test, col, signal, BUDGETS)
            ci = ratio_intervals(test, col, signal)
            rows.append(share.merge(ci, on=["budget", "group"]).assign(attribute=col, signal=name, model="XGBoost"))
    table = pd.concat(rows, ignore_index=True)
    rp.save_metrics(table, rp.TABLES_DIR / "equity_measured_share_xgboost.csv")

    age = table[table["attribute"] == "age_band"]
    with pd.option_context("display.width", 200):
        print(age.pivot_table(index=["signal", "budget"], columns="group", values="ratio_to_budget").round(2))
    log = table.assign(target=TARGET, model="XGBoost " + table["signal"] + " " + table["attribute"] + "="
                       + table["group"].astype(str) + " budget " + table["budget"].astype(str))
    write_results(log, task="27", seed=rp.RANDOM_STATE, metrics=["ratio_to_budget", "ratio_ci_low", "ratio_ci_high"],
                  notes=f"Table 3 for the deployed model: {N_BAGS} patient-bootstrap XGBoost models (spread) and "
                        f"calibrated XGBoost (closeness); {N_BOOT} patient resamples for intervals")
    print(f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
