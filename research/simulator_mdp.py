"""The measurement decision process for week 9 (ISEM 735 W9-1, results-log task 19).

State: what the quarterly policy loop (research/simulator_policy.py) shows a policy each quarter.
Action: measure (full panel) or not. Reward per quarter, as confirmed by Chinmay on 2026-10-26:

    r = -$214.49 x measurements - lambda x (months a lab-linked condition has truly begun but is undiagnosed)

with lambda in {$25, $50, $100, $200} per undiagnosed condition-month ($50 the headline; a judgement, not
a published price) and discounting at 3% a year (Sanders et al., 2016), 1.03^(-1/4) per quarter, over the
28 quarters of 2018-2024. Prevalent undiagnosed onsets count from January 2018.

The true value of a policy is its mean discounted return over the seed-42 evaluation cohort, computed exactly
from the simulator's truth. The logged dataset that week 9's off-policy estimators learn from comes from a
behavior policy: each quarter follow a quarterly guideline rule with probability 0.8, otherwise measure with
probability 0.25. The probability of measuring is therefore 0.85 when the guideline says measure and 0.05
when it does not, recorded with every decision.

    python -m research.simulator_mdp
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from research import simulator as sim
from research import simulator_policy as pol
from research import simulator_sweep as sweep
from research.cost_model import SIMULATOR_MEASUREMENT_COST

GAMMA = 1.03 ** -0.25
N_QUARTERS = sim.N_MONTHS // 3
LAMBDAS = [25.0, 50.0, 100.0, 200.0]
HEADLINE_LAMBDA = 50.0
FOLLOW_GUIDELINE = 0.8
EXPLORE_RATE = 0.25
QUARTER_DISCOUNT = GAMMA ** np.arange(N_QUARTERS)
# Discounted weight of months 0..m-1, each month weighted by its quarter's discount.
MONTH_WEIGHT = np.concatenate([[0.0], np.cumsum(GAMMA ** (np.arange(sim.N_MONTHS) // 3))])


def _undiagnosed_intervals(detected: pd.DataFrame) -> pd.DataFrame:
    d = detected.dropna(subset=["onset_month"])
    start = d["onset_month"].clip(lower=0).to_numpy()
    end = d["diagnosis_month"].fillna(sim.N_MONTHS).clip(upper=sim.N_MONTHS).to_numpy()
    keep = end > start
    return pd.DataFrame({"patient_id": d["patient_id"].to_numpy()[keep], "start": start[keep].astype(int),
                         "end": end[keep].astype(int)})


def patient_values(detected: pd.DataFrame, visits: pd.DataFrame, ids: pd.Index, lam: float) -> pd.Series:
    """Discounted return per patient."""
    months = sim.month_index(visits["test_date"])
    cost = pd.Series(SIMULATOR_MEASUREMENT_COST * GAMMA ** (months // 3), index=visits["patient_id"].to_numpy())
    cost = cost.groupby(level=0).sum().reindex(ids, fill_value=0.0)
    gaps = _undiagnosed_intervals(detected)
    weight = pd.Series(MONTH_WEIGHT[gaps["end"]] - MONTH_WEIGHT[gaps["start"]], index=gaps["patient_id"].to_numpy())
    weight = weight.groupby(level=0).sum().reindex(ids, fill_value=0.0)
    return -(cost + lam * weight)


def true_values(detected: pd.DataFrame, visits: pd.DataFrame, ids: pd.Index, lams=LAMBDAS) -> dict[float, float]:
    return {lam: float(patient_values(detected, visits, ids, lam).mean()) for lam in lams}


def quarter_components(detected: pd.DataFrame, visits: pd.DataFrame, ids: pd.Index) -> pd.DataFrame:
    """Per patient and quarter: measurements and undiagnosed condition-months (rewards for any lambda)."""
    grid = pd.MultiIndex.from_product([ids, np.arange(N_QUARTERS)], names=["patient_id", "quarter"])
    q = pd.Series(sim.month_index(visits["test_date"]) // 3, index=visits["patient_id"].to_numpy())
    n_visits = q.groupby([q.index, q.to_numpy()]).size().reindex(grid, fill_value=0)
    gaps = _undiagnosed_intervals(detected)
    lo = np.arange(N_QUARTERS) * 3
    overlap = np.clip(np.minimum(gaps["end"].to_numpy()[:, None], lo + 3)
                      - np.maximum(gaps["start"].to_numpy()[:, None], lo), 0, 3)
    months = pd.DataFrame(overlap, index=gaps["patient_id"].to_numpy()).groupby(level=0).sum()
    months = months.reindex(ids, fill_value=0).stack()
    months.index = months.index.set_names(["patient_id", "quarter"])
    out = pd.DataFrame({"visits": n_visits.to_numpy(), "undiagnosed_months": months.reindex(grid).to_numpy()},
                       index=grid).reset_index()
    return out


def guideline_actions(features: pd.DataFrame, phase: np.ndarray) -> np.ndarray:
    """Quarterly version of the W7-1 guideline: yearly once diabetes or kidney disease is known, otherwise
    every 3 years from 35 and every 5 years from 20; the first visit falls in the patient's phase month."""
    known = (features["known_type2_diabetes"] == 1) | (features["known_chronic_kidney_disease"] == 1)
    age = features["age"].to_numpy()
    interval = np.where(known, 12, np.where(age >= 35, 36, np.where(age >= 20, 60, np.inf)))
    first = (features["n_labs"].to_numpy() == 0) & (features["month"].to_numpy() >= phase) & np.isfinite(interval)
    due = (features["n_labs"].to_numpy() > 0) & (features["months_since_lab"].to_numpy() >= interval)
    return first | due


def guideline_phase(cohort: sweep.Cohort, seed: int) -> np.ndarray:
    ids = pd.Series(cohort.ages.index)
    u = sim.crn_uniform(sim.patient_number(ids), np.zeros(len(ids)), np.zeros(len(ids)), seed, stream=4)
    return 3 * np.floor(u * 4).astype(int)


def behavior_policy(cohort: sweep.Cohort, seed: int):
    """Guideline with probability 0.8, otherwise measure with probability 0.25 (returns action, propensity)."""
    phase = guideline_phase(cohort, seed)
    rng = np.random.default_rng(seed + 1000)

    def decide(features: pd.DataFrame):
        g = guideline_actions(features, phase)
        p = FOLLOW_GUIDELINE * g + (1 - FOLLOW_GUIDELINE) * EXPLORE_RATE
        return rng.random(len(features)) < p, p

    return decide


def guideline_policy(cohort: sweep.Cohort, seed: int):
    phase = guideline_phase(cohort, seed)

    def decide(features: pd.DataFrame):
        g = guideline_actions(features, phase)
        return g, g.astype(float)

    return decide


def logged_dataset(cohort: sweep.Cohort, source: dict[str, pd.DataFrame], params: sim.SimulatorParams, beta: float,
                   seed: int, static: pd.DataFrame | None = None, other: np.ndarray | None = None) -> pd.DataFrame:
    """One row per patient-quarter under the behavior policy: state, action, propensity, reward
    components and the next state's features."""
    run = pol.run_policy(cohort, source, params, beta, seed, policy_fn=behavior_policy(cohort, seed), record=True,
                         static=static, other=other)
    snaps = run.snapshots.assign(quarter=lambda s: s["month"] // 3)
    comp = quarter_components(run.detected, run.visits, cohort.ages.index)
    logged = snaps.merge(comp, on=["patient_id", "quarter"], how="left").sort_values(["patient_id", "quarter"])
    state_cols = pol.feature_columns(run.snapshots)
    nxt = logged.groupby("patient_id")[state_cols].shift(-1)
    logged = pd.concat([logged.reset_index(drop=True), nxt.add_prefix("next_").reset_index(drop=True)], axis=1)
    logged["terminal"] = logged["quarter"] == N_QUARTERS - 1
    return logged


def main() -> None:
    import argparse

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
    cohort = sweep.cohort_truth(source, params, args.patients, args.seed)
    static, other = pol.static_features(cohort, source), pol.other_dx_months(cohort, source)
    ids = cohort.ages.index
    out_dir = rp.PROJECT_ROOT / "data" / "processed" / "simulated" / "logged"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for beta in args.beta:
        runs = {"never measure": (cohort.observed.iloc[:0], None)}
        runs["observed care"] = (cohort.observed, None)
        for name, interval in sweep.FIXED_INTERVALS.items():
            runs[name] = (sweep.visits_from_months(sweep.fixed_schedule(pd.Series(ids), interval, args.seed)), None)
        for name, fn in [("quarterly guideline", guideline_policy(cohort, args.seed)),
                         ("behavior (guideline + 20% exploration)", behavior_policy(cohort, args.seed))]:
            run = pol.run_policy(cohort, source, params, beta, args.seed, policy_fn=fn, static=static, other=other)
            runs[name] = (run.visits, run.detected)
        for name, (visits, detected) in runs.items():
            if detected is None:
                detected = sweep.run_schedule(cohort, visits, params, beta, args.seed)
            values = true_values(detected, visits, ids)
            delay = sweep.delay_metrics(detected).set_index("condition").loc["all", "restricted_mean_delay"]
            rows.append({"beta": beta, "schedule": name,
                         "measurements_per_patient_year": len(visits) / len(ids) / sweep.YEARS,
                         "restricted_mean_delay": delay,
                         **{f"value_lambda_{lam:g}": v for lam, v in values.items()}})
        logged = logged_dataset(cohort, source, params, beta, args.seed, static=static, other=other)
        logged.to_parquet(out_dir / f"logged_beta_{beta:g}.parquet", index=False)
        print(f"beta={beta}: {len(logged):,} logged transitions, measure rate {logged['action'].mean():.3f} | "
              f"{time.perf_counter() - start:.0f}s")
    table = pd.DataFrame(rows)
    rp.save_metrics(table, rp.TABLES_DIR / "simulator_true_values.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(table.round(2).to_string(index=False))
    log = table.assign(target="discounted return per patient", model="beta=" + table["beta"].map("{:g}".format)
                       + " " + table["schedule"])
    write_results(log, task="19", seed=args.seed,
                  metrics=[f"value_lambda_{lam:g}" for lam in LAMBDAS] + ["measurements_per_patient_year"],
                  notes=f"true policy values in the simulator, {args.patients:,} patients; reward -$214.49 per "
                        "measurement - lambda per undiagnosed condition-month; 3%/yr discount")
    print(f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
