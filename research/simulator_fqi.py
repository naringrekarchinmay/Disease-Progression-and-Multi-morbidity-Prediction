"""Week 9, part 2: a learned measurement policy by fitted Q-iteration (ISEM 735 W9-2, results-log task 20).

The policy is learned offline, the way it would be from real records: from logged data of a separate
30,000-patient cohort (seed 43) cared for by the behavior policy of research/simulator_mdp.py (guideline
with 20% exploration). The horizon is finite (28 quarters) and the calendar month is part of the state, so
fitted Q-iteration runs as backward induction: for quarter t, regress r_t + gamma * max_a Q_{t+1}(s', a)
on the state, separately for each action, using only that quarter's rows. Rewards follow simulator_mdp:
-$214.49 per measurement - lambda per undiagnosed condition-month, 3% a year discounting.

The greedy policy is then run on the seed-42 evaluation cohort and its true value computed exactly, next
to the fixed schedules, the guideline, the behavior policy and an oracle that knows every onset and
measures once at the first quarter after it (a bound no real policy can reach). Lambda is $200 (headline)
and $400, plus $100 as a low-price check.

    python -m research.simulator_fqi

This module loads XGBoost, so it must never import torch (see research/gru_model.py).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from research import simulator as sim
from research import simulator_mdp as mdp
from research import simulator_policy as pol
from research import simulator_sweep as sweep
from research.cost_model import SIMULATOR_MEASUREMENT_COST

LEARN_LAMBDAS = [100.0, 200.0, 400.0]
TRAIN_PATIENTS = 30000
XGB_REGRESSOR = dict(n_estimators=150, max_depth=4, learning_rate=0.1, min_child_weight=20, subsample=0.9,
                     tree_method="hist")


def reward(logged: pd.DataFrame, lam: float) -> np.ndarray:
    return (-SIMULATOR_MEASUREMENT_COST * logged["visits"] - lam * logged["undiagnosed_months"]).to_numpy(dtype=float)


def q_values(models: tuple, states: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return models[0].predict(states), models[1].predict(states)


def fit_fqi(logged: pd.DataFrame, lam: float, state_cols: list[str], make_regressor,
            n_quarters: int = mdp.N_QUARTERS, gamma: float = mdp.GAMMA) -> list[tuple]:
    """Backward induction: models[t] = (Q_t(., measure=0), Q_t(., measure=1))."""
    models: list[tuple | None] = [None] * n_quarters
    next_cols = [f"next_{c}" for c in state_cols]
    for t in range(n_quarters - 1, -1, -1):
        rows = logged[logged["quarter"] == t]
        target = reward(rows, lam)
        if t < n_quarters - 1:
            nxt = rows[next_cols].set_axis(state_cols, axis=1)
            q0, q1 = q_values(models[t + 1], nxt)
            target = target + gamma * np.maximum(q0, q1)
        pair = []
        for a in (0, 1):
            mask = (rows["action"] == a).to_numpy()
            model = make_regressor()
            model.fit(rows.loc[mask, state_cols], target[mask])
            pair.append(model)
        models[t] = tuple(pair)
    return models


def greedy_policy(models: list[tuple], state_cols: list[str]):
    def decide(features: pd.DataFrame):
        quarter = int(features["month"].iloc[0]) // 3
        q0, q1 = q_values(models[quarter], features[state_cols])
        act = q1 > q0
        return act, act.astype(float)

    return decide


def oracle_visits(cohort: sweep.Cohort) -> pd.DataFrame:
    """One visit at the first decision month on or after each lab-linked onset inside 2018-2024."""
    t = cohort.truth[cohort.truth["onset_month"] < sim.N_MONTHS]
    first = (np.ceil(t["onset_month"].clip(lower=0) / 3) * 3).astype(int)
    months = pd.DataFrame({"patient_id": t["patient_id"].to_numpy(), "month": first.to_numpy()})
    months = months[months["month"] < sim.N_MONTHS].drop_duplicates().sort_values(["patient_id", "month"])
    return sweep.visits_from_months(months)


def main() -> None:
    import argparse

    from xgboost import XGBRegressor

    import run_pipeline as rp
    from research.calibration import write_results

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patients", type=int, default=100000)
    parser.add_argument("--beta", type=float, nargs="+", default=sweep.BETAS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.perf_counter()
    source = sim.load_source()
    params = sim.fit_params(source)
    train_c = sweep.cohort_truth(source, params, TRAIN_PATIENTS, args.seed + 1)
    train_s, train_o = pol.static_features(train_c, source), pol.other_dx_months(train_c, source)
    eval_c = sweep.cohort_truth(source, params, args.patients, args.seed)
    eval_s, eval_o = pol.static_features(eval_c, source), pol.other_dx_months(eval_c, source)
    ids = eval_c.ages.index
    reference = pd.read_csv(rp.TABLES_DIR / "simulator_true_values.csv")
    make = lambda: XGBRegressor(**XGB_REGRESSOR, random_state=args.seed)  # noqa: E731
    print(f"cohorts ready | {time.perf_counter() - start:.0f}s")

    rows = []
    for beta in args.beta:
        logged = mdp.logged_dataset(train_c, source, params, beta, args.seed + 1, static=train_s, other=train_o)
        state_cols = [c for c in pol.feature_columns(logged) if c in logged.columns and f"next_{c}" in logged.columns]
        oracle = oracle_visits(eval_c)
        oracle_detected = sweep.run_schedule(eval_c, oracle, params, beta, args.seed)
        oracle_values = mdp.true_values(oracle_detected, oracle, ids, mdp.LAMBDAS)
        for lam in LEARN_LAMBDAS:
            models = fit_fqi(logged, lam, state_cols, make)
            run = pol.run_policy(eval_c, source, params, beta, args.seed, policy_fn=greedy_policy(models, state_cols),
                                 static=eval_s, other=eval_o)
            values = mdp.true_values(run.detected, run.visits, ids, mdp.LAMBDAS)
            summary = pd.DataFrame(sweep.summarize(f"FQI policy (lambda {lam:g})", beta, eval_c, run.visits,
                                                   run.detected, run.labs)).set_index("condition").loc["all"]
            ref = reference[reference["beta"] == beta].set_index("schedule")[f"value_lambda_{lam:g}"]
            q0, q1 = q_values(models[0], logged.loc[logged["quarter"] == 0, state_cols])
            rows.append({
                "beta": beta, "lambda": lam, "schedule": f"FQI policy (lambda {lam:g})",
                "measurements_per_patient_year": summary["measurements_per_patient_year"],
                "restricted_mean_delay": summary["restricted_mean_delay"],
                "true_value": values[lam],
                "fqi_estimate_start_value": float(np.maximum(q0, q1).mean()),
                "oracle_value": oracle_values[lam],
                "best_reference": ref.idxmax(), "best_reference_value": ref.max(),
                "guideline_value": ref["quarterly guideline"], "behavior_value": ref["behavior (guideline + 20% exploration)"],
                "never_value": ref["never measure"], "twelve_month_value": ref["12-month"],
                **{f"measurement_ratio_age_{b}": summary[f"measurement_ratio_age_{b}"] for b in sweep.AGE_LABELS},
            })
            r = rows[-1]
            print(f"beta={beta} lambda={lam:g}: FQI true {r['true_value']:.0f} (own estimate "
                  f"{r['fqi_estimate_start_value']:.0f}) vs best reference {r['best_reference']} "
                  f"{r['best_reference_value']:.0f}, oracle {r['oracle_value']:.0f}; "
                  f"{r['measurements_per_patient_year']:.2f}/yr, delay {r['restricted_mean_delay']:.1f} | "
                  f"{time.perf_counter() - start:.0f}s")
    table = pd.DataFrame(rows)
    rp.save_metrics(table, rp.TABLES_DIR / "simulator_fqi_policy.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(table.drop(columns=[c for c in table.columns if c.startswith("measurement_ratio")]).round(1)
              .to_string(index=False))
    log = table.assign(target="discounted return per patient",
                       model="beta=" + table["beta"].map("{:g}".format) + " " + table["schedule"])
    write_results(log, task="20", seed=args.seed,
                  metrics=["true_value", "fqi_estimate_start_value", "best_reference_value", "oracle_value",
                           "measurements_per_patient_year"],
                  notes=f"fitted Q-iteration (backward induction, XGBoost) on {TRAIN_PATIENTS:,} logged behavior-policy "
                        f"patients (seed 43); true value on {args.patients:,} evaluation patients")
    print(f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
