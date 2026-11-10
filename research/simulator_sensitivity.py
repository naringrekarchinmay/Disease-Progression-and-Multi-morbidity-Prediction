"""The pre-registered sensitivity sweep (ISEM 735, task 26; plan in research/preregistration.md).

Settings, one factor at a time around the base case, each crossed with every beta in {0, 0.25, 0.5, 1, 2}:
the onset window W scaled 0.5x / 1x / 1.5x, and the clinical lag ranges scaled 0.5x / 1x / 1.5x (the
hyperlipidemia rule is left alone). Each setting runs for replicates r = 0, 1, 2 with evaluation seed
42 + 1000r and training seed 43 + 1000r; r = 0 reproduces task 20.

Every run scores the fixed policies (never measure, observed care, quarterly guideline, 12-month, 3-month)
and the fitted Q-iteration policy retrained at lambda $100, $200 and $400 (task 20's code and settings).
Per evaluation patient it saves the discounted measurement cost C and the discounted undiagnosed
condition-months U, with and without hyperlipidemia, so a policy's value at any lambda is -C - lambda * U.
`--analyze` pools replicates and applies the pre-registered decision rules.

    python -m research.simulator_sensitivity            # run every missing (setting, replicate)
    python -m research.simulator_sensitivity --analyze

This module loads XGBoost, so it must never import torch.
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
import pandas as pd

import run_pipeline as rp
from research import simulator as sim
from research import simulator_mdp as mdp
from research import simulator_policy as pol
from research import simulator_sweep as sweep
from research.simulator_fqi import LEARN_LAMBDAS, TRAIN_PATIENTS, XGB_REGRESSOR, fit_fqi, greedy_policy
from research.stats_ci import ALPHA

OUT_DIR = rp.PROJECT_ROOT / "data" / "processed" / "simulated" / "sensitivity"
FACTORS = {"base": (1.0, 1.0), "W x0.5": (0.5, 1.0), "W x1.5": (1.5, 1.0),
           "lag x0.5": (1.0, 0.5), "lag x1.5": (1.0, 1.5)}  # (window multiplier, lag multiplier)
REPLICATES = [0, 1, 2]
FIXED = ["never measure", "observed care", "quarterly guideline", "12-month", "3-month"]
EXCLUDED = "hyperlipidemia"
N_BOOT = 2000


def scaled_params(params: sim.SimulatorParams, window_mult: float, lag_mult: float) -> sim.SimulatorParams:
    window = {c: max(1, int(round(w * window_mult))) for c, w in params.window.items()}
    lag = {c: None if b is None else (max(1, int(round(b[0] * lag_mult))), max(1, int(round(b[1] * lag_mult))))
           for c, b in params.clinical_lag.items()}
    return dataclasses.replace(params, window=window, clinical_lag=lag)


def components(detected: pd.DataFrame, visits: pd.DataFrame, ids: pd.Index) -> dict[str, np.ndarray]:
    """Per-patient discounted cost C and undiagnosed condition-months U (all, and without hyperlipidemia)."""
    cost = -mdp.patient_values(detected, visits, ids, 0.0).to_numpy()
    u_all = -mdp.patient_values(detected, visits, ids, 1.0).to_numpy() - cost
    rest = detected[detected["condition"] != EXCLUDED]
    u_rest = -mdp.patient_values(rest, visits, ids, 1.0).to_numpy() - cost
    return {"C": cost.astype(np.float32), "U": u_all.astype(np.float32), "U_nolipid": u_rest.astype(np.float32)}


def run_path(factor: str, replicate: int, beta: float, out_dir=None):
    tag = factor.replace(" ", "").replace(".", "p")
    return (out_dir or OUT_DIR) / f"{tag}_r{replicate}_beta{beta:g}.parquet"


def run_setting(source, base_params, factor: str, replicate: int, n_eval: int, betas, log=print,
                out_dir=None, n_train: int = TRAIN_PATIENTS) -> None:
    from xgboost import XGBRegressor

    params = scaled_params(base_params, *FACTORS[factor])
    eval_seed, train_seed = 42 + 1000 * replicate, 43 + 1000 * replicate
    todo = [b for b in betas if not run_path(factor, replicate, b, out_dir).exists()]
    if not todo:
        return
    start = time.perf_counter()
    train_c = sweep.cohort_truth(source, params, n_train, train_seed)
    train_s, train_o = pol.static_features(train_c, source), pol.other_dx_months(train_c, source)
    eval_c = sweep.cohort_truth(source, params, n_eval, eval_seed)
    eval_s, eval_o = pol.static_features(eval_c, source), pol.other_dx_months(eval_c, source)
    ids = eval_c.ages.index
    make = lambda: XGBRegressor(**XGB_REGRESSOR, random_state=eval_seed)  # noqa: E731
    for beta in todo:
        out, summary = {}, []

        def record(name, visits, detected):
            for key, arr in components(detected, visits, ids).items():
                out[f"{name}|{key}"] = arr
            delay = sweep.delay_metrics(detected).set_index("condition").loc["all", "restricted_mean_delay"]
            summary.append({"policy": name, "measurements_per_patient_year": len(visits) / len(ids) / sweep.YEARS,
                            "restricted_mean_delay": delay})

        record("never measure", eval_c.observed.iloc[:0], sweep.run_schedule(eval_c, eval_c.observed.iloc[:0], params,
                                                                             beta, eval_seed))
        record("observed care", eval_c.observed, sweep.run_schedule(eval_c, eval_c.observed, params, beta, eval_seed))
        for name, interval in [("12-month", 12), ("3-month", 3)]:
            visits = sweep.visits_from_months(sweep.fixed_schedule(pd.Series(ids), interval, eval_seed))
            record(name, visits, sweep.run_schedule(eval_c, visits, params, beta, eval_seed))
        run = pol.run_policy(eval_c, source, params, beta, eval_seed, policy_fn=mdp.guideline_policy(eval_c, eval_seed),
                             static=eval_s, other=eval_o)
        record("quarterly guideline", run.visits, run.detected)

        logged = mdp.logged_dataset(train_c, source, params, beta, train_seed, static=train_s, other=train_o)
        state_cols = [c for c in pol.feature_columns(logged) if c in logged.columns and f"next_{c}" in logged.columns]
        for lam in LEARN_LAMBDAS:
            models = fit_fqi(logged, lam, state_cols, make)
            run = pol.run_policy(eval_c, source, params, beta, eval_seed,
                                 policy_fn=greedy_policy(models, state_cols), static=eval_s, other=eval_o)
            record(f"FQI {lam:g}", run.visits, run.detected)

        frame = pd.DataFrame(out)
        frame.attrs = {}
        path = run_path(factor, replicate, beta, out_dir)
        frame.to_parquet(path, index=False)
        pd.DataFrame(summary).assign(factor=factor, replicate=replicate, beta=beta).to_csv(
            path.with_suffix(".csv"), index=False)
        log(f"{factor} r{replicate} beta={beta:g} done | {time.perf_counter() - start:.0f}s")


# --- analysis ----------------------------------------------------------------------------------------


def values(frame: pd.DataFrame, policy: str, lam: float, lipid: bool = True) -> np.ndarray:
    u = frame[f"{policy}|U" if lipid else f"{policy}|U_nolipid"].to_numpy(np.float64)
    return -frame[f"{policy}|C"].to_numpy(np.float64) - lam * u


def primary_outcome(frames: list[pd.DataFrame], lam: float, lipid: bool, seed: int = 42) -> dict:
    """Learned minus best fixed policy, pooled over replicates; patient-level bootstrap of the pooled mean."""
    pooled = {p: np.concatenate([values(f, p, lam, lipid) for f in frames]) for p in FIXED + [f"FQI {lam:g}"]}
    best = max(FIXED, key=lambda p: pooled[p].mean())
    diff = pooled[f"FQI {lam:g}"] - pooled[best]
    rng = np.random.default_rng(seed)
    n = len(diff)
    draws = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    per_rep = [values(f, f"FQI {lam:g}", lam, lipid).mean() - values(f, best, lam, lipid).mean() for f in frames]
    return {"best_fixed": best, "learned_minus_best": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "replicate_sd": float(np.std(per_rep, ddof=1)) if len(per_rep) > 1 else np.nan,
            "learned_value": float(pooled[f"FQI {lam:g}"].mean()), "best_fixed_value": float(pooled[best].mean()),
            "adaptive_wins": bool(lo > 0), "n_replicates": len(frames)}


def break_even(frames: list[pd.DataFrame], policy: str) -> float:
    """Lambda above which `policy` beats never measuring: C / (U_never - U_policy)."""
    c = np.mean([f[f"{policy}|C"].mean() for f in frames])
    gain = np.mean([f["never measure|U"].mean() - f[f"{policy}|U"].mean() for f in frames])
    return float(c / gain) if gain > 0 else np.inf


def analyze() -> pd.DataFrame:
    rows, summaries = [], []
    for factor in FACTORS:
        for beta in sweep.BETAS:
            paths = [run_path(factor, r, beta) for r in REPLICATES if run_path(factor, r, beta).exists()]
            if not paths:
                continue
            frames = [pd.read_parquet(p) for p in paths]
            summaries += [pd.read_csv(p.with_suffix(".csv")) for p in paths]
            for lam in LEARN_LAMBDAS:
                for lipid in (True, False):
                    rows.append({"factor": factor, "beta": beta, "lambda": lam,
                                 "reward": "all conditions" if lipid else "without hyperlipidemia",
                                 **primary_outcome(frames, lam, lipid)})
            for p in FIXED[1:]:
                rows.append({"factor": factor, "beta": beta, "lambda": np.nan, "reward": "break-even vs never",
                             "best_fixed": p, "learned_minus_best": break_even(frames, p),
                             "n_replicates": len(frames)})
    table = pd.DataFrame(rows)
    rp.save_metrics(table, rp.TABLES_DIR / "simulator_sensitivity.csv")
    secondary = (pd.concat(summaries).groupby(["factor", "beta", "policy"], sort=False)
                 [["measurements_per_patient_year", "restricted_mean_delay"]].mean().reset_index())
    rp.save_metrics(secondary, rp.TABLES_DIR / "simulator_sensitivity_secondary.csv")
    return table


def robust_region(table: pd.DataFrame, reward: str = "all conditions") -> pd.DataFrame:
    """Pre-registered rule 2: adaptive wins at every W and lag level for that beta and lambda."""
    t = table[table["reward"] == reward]
    grid = t.pivot_table(index="beta", columns="lambda", values="adaptive_wins", aggfunc="all")
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patients", type=int, default=100000)
    parser.add_argument("--factors", nargs="+", default=list(FACTORS))
    parser.add_argument("--replicates", type=int, nargs="+", default=REPLICATES)
    parser.add_argument("--beta", type=float, nargs="+", default=sweep.BETAS)
    parser.add_argument("--train-patients", type=int, default=TRAIN_PATIENTS)
    parser.add_argument("--out", default=None, help="Output folder (default: the pre-registered sweep's).")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        table = analyze()
        main_rows = table[table["reward"] != "break-even vs never"]
        with pd.option_context("display.width", 250, "display.max_rows", 400):
            print(main_rows.drop(columns=["learned_value", "best_fixed_value"]).round(1).to_string(index=False))
            for reward in ("all conditions", "without hyperlipidemia"):
                print(f"\nRobust region ({reward}): adaptive wins at every W and lag level")
                print(robust_region(table, reward).to_string())
        return

    out_dir = rp.PROJECT_ROOT / args.out if args.out else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source = sim.load_source()
    params = sim.fit_params(source)
    for r in args.replicates:
        for factor in args.factors:
            run_setting(source, params, factor, r, args.patients, args.beta,
                        log=lambda m: print(f"{m} | total {time.perf_counter() - start:.0f}s", flush=True),
                        out_dir=out_dir, n_train=args.train_patients)


if __name__ == "__main__":
    main()
