"""RQ4 in the simulator: does a learned policy amplify existing under-observation? (ISEM 735, task 28)

The Kaggle data has no subgroup that is measured less (research/equity_notes.md, section 1), so the gap is
injected. Chinmay's choices (2026-11-09): the subgroup is Medicaid and uninsured patients (28%); in the history
the policies learn from they are measured at rho = 1 (control), 0.5 or 0.25 times everyone else's rate; the
setting is the sweep's headline, beta 1 and lambda $200, with three replicates (evaluation seed 42 + 1000r,
training seed 43 + 1000r).

Under-observation enters in two places, as it would in real records:
- the logged history: the behavior policy of research/simulator_mdp.py (guideline with 20% exploration) has
  its probability of measuring multiplied by rho for the subgroup; fitted Q-iteration learns from these logs;
- observed care: each of the subgroup's copied Kaggle visits is kept with probability rho.

The true disease process is unchanged. Policies scored on the evaluation cohort: the under-observed history
itself (the behavior policy run there), observed care, the guideline, the 12-month schedule, and the learned
policy. Outcomes, fixed before running: for the subgroup and the rest, measurements per patient-year and
restricted mean detection delay; the subgroup-to-rest measurement ratio, the delay gap (subgroup minus rest)
and value per patient at $200. The learned policy **amplifies** the gap if its subgroup-to-rest measurement
ratio is lower than the history's, or its delay gap larger than at rho = 1, by more than a replicate t interval.

    python -m research.simulator_equity --replicates 0      # one replicate per process
    python -m research.simulator_equity --analyze

This module loads XGBoost, so it must never import torch.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

import run_pipeline as rp
from research import simulator as sim
from research import simulator_mdp as mdp
from research import simulator_policy as pol
from research import simulator_sweep as sweep
from research.simulator_fqi import TRAIN_PATIENTS, XGB_REGRESSOR, fit_fqi, greedy_policy

SUBGROUP = ("medicaid", "uninsured")
SUBGROUP_CODES = [pol.INSURANCE[s] for s in SUBGROUP]
RHOS = [1.0, 0.5, 0.25]
BETA = 1.0
LAMBDA = 200.0
REPLICATES = [0, 1, 2]
OUT_DIR = rp.PROJECT_ROOT / "data" / "processed" / "simulated" / "equity"


def in_subgroup(features: pd.DataFrame) -> np.ndarray:
    return features["insurance"].isin(SUBGROUP_CODES).to_numpy()


def under_observed_behavior(cohort: sweep.Cohort, seed: int, rho: float):
    """mdp.behavior_policy with the subgroup's probability of measuring multiplied by rho.

    At rho = 1 it draws exactly the same decisions as mdp.behavior_policy (same phase, same random stream).
    """
    phase = mdp.guideline_phase(cohort, seed)
    rng = np.random.default_rng(seed + 1000)

    def decide(features: pd.DataFrame):
        g = mdp.guideline_actions(features, phase)
        p = mdp.FOLLOW_GUIDELINE * g + (1 - mdp.FOLLOW_GUIDELINE) * mdp.EXPLORE_RATE
        p = p * np.where(in_subgroup(features), rho, 1.0)
        return rng.random(len(features)) < p, p

    return decide


def thin_visits(visits: pd.DataFrame, subgroup_ids: set, rho: float, seed: int) -> pd.DataFrame:
    """Keep each subgroup visit with probability rho; other patients' visits are untouched."""
    if rho >= 1:
        return visits
    rng = np.random.default_rng(seed + 2000)
    sub = visits["patient_id"].isin(subgroup_ids).to_numpy()
    keep = ~sub | (rng.random(len(visits)) < rho)
    return visits[keep].reset_index(drop=True)


def group_metrics(name: str, visits: pd.DataFrame, detected: pd.DataFrame, ids: pd.Index, sub: pd.Series) -> dict:
    values = mdp.patient_values(detected, visits, ids, LAMBDA)
    counts = visits["patient_id"].value_counts().reindex(ids, fill_value=0)
    out = {"policy": name}
    for label, mask in [("subgroup", sub), ("rest", ~sub)]:
        members = set(ids[mask.to_numpy()])
        out[f"measurements_{label}"] = counts[mask.to_numpy()].mean() / sweep.YEARS
        part = detected[detected["patient_id"].isin(members)]
        out[f"delay_{label}"] = sweep.delay_metrics(part).set_index("condition").loc["all", "restricted_mean_delay"]
        out[f"value_{label}"] = values[mask.to_numpy()].mean()
    out["measurement_ratio"] = out["measurements_subgroup"] / out["measurements_rest"]
    out["delay_gap"] = out["delay_subgroup"] - out["delay_rest"]
    return out


def run_replicate(source, params, replicate: int, n_eval: int, rhos=RHOS, log=print) -> None:
    from xgboost import XGBRegressor

    eval_seed, train_seed = 42 + 1000 * replicate, 43 + 1000 * replicate
    start = time.perf_counter()
    train_c = sweep.cohort_truth(source, params, TRAIN_PATIENTS, train_seed)
    train_s, train_o = pol.static_features(train_c, source), pol.other_dx_months(train_c, source)
    eval_c = sweep.cohort_truth(source, params, n_eval, eval_seed)
    eval_s, eval_o = pol.static_features(eval_c, source), pol.other_dx_months(eval_c, source)
    ids = eval_c.ages.index
    sub = pd.Series(eval_s["insurance"].isin(SUBGROUP_CODES).to_numpy(), index=ids)
    sub_ids = set(ids[sub.to_numpy()])
    make = lambda: XGBRegressor(**XGB_REGRESSOR, random_state=eval_seed)  # noqa: E731

    fixed = []
    run = pol.run_policy(eval_c, source, params, BETA, eval_seed, policy_fn=mdp.guideline_policy(eval_c, eval_seed),
                         static=eval_s, other=eval_o)
    fixed.append(group_metrics("quarterly guideline", run.visits, run.detected, ids, sub))
    visits = sweep.visits_from_months(sweep.fixed_schedule(pd.Series(ids), 12, eval_seed))
    fixed.append(group_metrics("12-month", visits, sweep.run_schedule(eval_c, visits, params, BETA, eval_seed), ids, sub))

    for rho in rhos:
        path = OUT_DIR / f"r{replicate}_rho{rho:g}.csv"
        if path.exists():
            continue
        rows = list(fixed)
        observed = thin_visits(eval_c.observed, sub_ids, rho, eval_seed)
        rows.append(group_metrics("observed care", observed,
                                  sweep.run_schedule(eval_c, observed, params, BETA, eval_seed), ids, sub))
        run = pol.run_policy(eval_c, source, params, BETA, eval_seed,
                             policy_fn=under_observed_behavior(eval_c, eval_seed, rho), static=eval_s, other=eval_o)
        rows.append(group_metrics("history (behavior policy)", run.visits, run.detected, ids, sub))
        logged = mdp.logged_dataset(train_c, source, params, BETA, train_seed, static=train_s, other=train_o,
                                    policy_fn=under_observed_behavior(train_c, train_seed, rho))
        state_cols = [c for c in pol.feature_columns(logged) if c in logged.columns and f"next_{c}" in logged.columns]
        models = fit_fqi(logged, LAMBDA, state_cols, make)
        run = pol.run_policy(eval_c, source, params, BETA, eval_seed, policy_fn=greedy_policy(models, state_cols),
                             static=eval_s, other=eval_o)
        rows.append(group_metrics("learned (FQI $200)", run.visits, run.detected, ids, sub))
        pd.DataFrame(rows).assign(rho=rho, replicate=replicate, subgroup_share=sub.mean()).to_csv(path, index=False)
        log(f"r{replicate} rho={rho:g} done | {time.perf_counter() - start:.0f}s")


def replicate_interval(x: pd.Series) -> tuple[float, float, float]:
    x = x.dropna()
    mean, n = float(x.mean()), len(x)
    if n < 2:
        return mean, np.nan, np.nan
    half = float(t_dist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return mean, mean - half, mean + half


def analyze() -> pd.DataFrame:
    runs = pd.concat([pd.read_csv(p) for p in sorted(OUT_DIR.glob("r*_rho*.csv"))], ignore_index=True)
    rows = []
    for (policy, rho), part in runs.groupby(["policy", "rho"], sort=False):
        row = {"policy": policy, "rho": rho, "n_replicates": len(part)}
        for metric in ["measurements_subgroup", "measurements_rest", "measurement_ratio", "delay_subgroup",
                       "delay_rest", "delay_gap", "value_subgroup", "value_rest"]:
            row[metric], row[f"{metric}_low"], row[f"{metric}_high"] = replicate_interval(part[metric])
        rows.append(row)
    summary = pd.DataFrame(rows)

    # Amplification, paired within replicate: learned vs history ratio, and learned delay gap vs rho = 1.
    amp = []
    for rho, part in runs.groupby("rho"):
        learned = part[part["policy"] == "learned (FQI $200)"].set_index("replicate")
        history = part[part["policy"] == "history (behavior policy)"].set_index("replicate")
        control = runs[(runs["rho"] == 1.0) & (runs["policy"] == "learned (FQI $200)")].set_index("replicate")
        ratio_change = learned["measurement_ratio"] - history["measurement_ratio"]
        gap_change = learned["delay_gap"] - control["delay_gap"]
        for name, series in [("learned ratio minus history ratio", ratio_change),
                             ("learned delay gap minus rho=1 learned delay gap", gap_change)]:
            mean, lo, hi = replicate_interval(series)
            amp.append({"rho": rho, "comparison": name, "estimate": mean, "low": lo, "high": hi})
    amplification = pd.DataFrame(amp)
    rp.save_metrics(summary, rp.TABLES_DIR / "simulator_equity.csv")
    rp.save_metrics(amplification, rp.TABLES_DIR / "simulator_equity_amplification.csv")
    return summary, amplification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patients", type=int, default=100000)
    parser.add_argument("--replicates", type=int, nargs="+", default=REPLICATES)
    parser.add_argument("--rho", type=float, nargs="+", default=RHOS)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        summary, amplification = analyze()
        show = ["policy", "rho", "measurements_subgroup", "measurements_rest", "measurement_ratio",
                "measurement_ratio_low", "measurement_ratio_high", "delay_subgroup", "delay_rest", "delay_gap",
                "delay_gap_low", "delay_gap_high"]
        with pd.option_context("display.width", 250, "display.max_rows", 100):
            print(summary[show].round(3).to_string(index=False))
            print(amplification.round(3).to_string(index=False))
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source = sim.load_source()
    params = sim.fit_params(source)
    for r in args.replicates:
        run_replicate(source, params, r, args.patients, args.rho,
                      log=lambda m: print(f"{m} | total {time.perf_counter() - start:.0f}s", flush=True))


if __name__ == "__main__":
    main()
