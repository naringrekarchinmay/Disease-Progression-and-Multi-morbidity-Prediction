"""Week 9, part 3: off-policy evaluation checked against the simulator's truth (ISEM 735 W9-3, task 21).

The proposal commits to fitted Q-evaluation (FQE) as its off-policy estimator. Here FQE, weighted
per-decision importance sampling (WPDIS) and weighted doubly robust (WDR) estimates are computed from the
seed-42 logged dataset of research/simulator_mdp.py: 100,000 patients, 28 quarters, cared for by the behavior
policy (guideline with 20% exploration; propensity of measuring 0.85 or 0.05, recorded). Each estimate is
compared with the target policy's true value, computed exactly in the simulator on the same patients.

Estimators (gamma = 0.9926 per quarter):
- FQE (Le, Voloshin & Yue, 2019): backward induction on the logged rows,
  Q_t(s, a) <- r_t + gamma * E_{a' ~ pi}[Q_{t+1}(s', a')], one XGBoost per action and quarter;
  the estimate is the mean of E_pi[Q_0(s_0, .)].
- WPDIS: per-decision importance weights w_t = prod_{k<=t} pi(a_k|s_k) / b(a_k|s_k), normalized across
  patients at each quarter (Thomas & Brunskill, 2016).
- WDR: the weighted doubly robust estimator of Thomas and Brunskill (2016), which builds on Jiang and Li
  (2016), using FQE's Q as the model.
Percentile intervals for WPDIS and WDR come from 200 bootstrap resamples of patients (FQE's model is held
fixed in the WDR bootstrap, and FQE itself is not bootstrapped).

Targets: never measure, the quarterly guideline, the behavior policy itself (an on-policy check), and the
fitted Q-iteration policies of task 20 at lambda $200 and $400, retrained exactly as there.

    python -m research.simulator_ope

This module loads XGBoost, so it must never import torch (see research/gru_model.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research import simulator_mdp as mdp

N_BOOT = 200
XGB_REGRESSOR = dict(n_estimators=80, max_depth=4, learning_rate=0.1, min_child_weight=20, subsample=0.9,
                     tree_method="hist")
EVAL_LAMBDAS = [200.0, 400.0]


@dataclass
class FQEFit:
    value: float
    q_taken: np.ndarray  # Q_t(s_t, a_t) for every logged row
    v_state: np.ndarray  # E_pi[Q_t(s_t, .)] for every logged row


def _patient_codes(df: pd.DataFrame) -> np.ndarray:
    if "_pcode" in df.columns:  # cached by main(), since the bootstrap calls this hundreds of times
        return df["_pcode"].to_numpy()
    return pd.factorize(df["patient_id"], sort=True)[0]


def cumulative_weights(df: pd.DataFrame, target_p: np.ndarray) -> np.ndarray:
    """prod_{k<=t} pi(a_k|s_k)/b(a_k|s_k) along each patient's quarters (df sorted by patient, quarter)."""
    a = df["action"].to_numpy() == 1
    b = df["p_measure"].to_numpy(dtype=float)
    p = np.asarray(target_p, dtype=float)
    rho = np.where(a, p / b, (1 - p) / (1 - b))
    return pd.Series(rho).groupby(_patient_codes(df)).cumprod().to_numpy()


def _row_counts(df: pd.DataFrame, patients) -> np.ndarray:
    if patients is None:
        return np.ones(len(df))
    return np.asarray(patients, dtype=float)[_patient_codes(df)]


def _per_quarter_normalized(df: pd.DataFrame, w: np.ndarray, counts: np.ndarray) -> np.ndarray:
    q = df["quarter"].to_numpy()
    cw = counts * w
    totals = np.bincount(q, weights=cw)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = cw / totals[q]
    return np.nan_to_num(out)  # a quarter with no matching trajectory contributes nothing


def wpdis(df: pd.DataFrame, r: np.ndarray, w: np.ndarray, gamma: float = mdp.GAMMA, patients=None) -> float:
    counts = _row_counts(df, patients)
    norm = _per_quarter_normalized(df, w, counts)
    return float(np.sum(norm * r * gamma ** df["quarter"].to_numpy()))


def previous_weights(df: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    return pd.Series(w).groupby(_patient_codes(df)).shift(1).fillna(1.0).to_numpy()


def wdr(df: pd.DataFrame, r: np.ndarray, w: np.ndarray, q_taken: np.ndarray, v_state: np.ndarray,
        gamma: float = mdp.GAMMA, patients=None, w_prev: np.ndarray | None = None) -> float:
    counts = _row_counts(df, patients)
    w_prev = previous_weights(df, w) if w_prev is None else w_prev
    norm = _per_quarter_normalized(df, w, counts)
    norm_prev = _per_quarter_normalized(df, w_prev, counts)
    disc = gamma ** df["quarter"].to_numpy()
    return float(np.sum(disc * (norm * (r - q_taken) + norm_prev * v_state)))


def ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.sum(w**2)) if np.sum(w**2) > 0 else 0.0


def bootstrap(fn, df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = 42) -> tuple[float, float]:
    """95% percentile interval of fn(patient resample counts)."""
    n = df["patient_id"].nunique()
    rng = np.random.default_rng(seed)
    draws = [fn(rng.multinomial(n, np.full(n, 1 / n))) for _ in range(n_boot)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def fqe(df: pd.DataFrame, r: np.ndarray, pi, state_cols: list[str], make_regressor,
        n_quarters: int = mdp.N_QUARTERS, gamma: float = mdp.GAMMA) -> FQEFit:
    """Fitted Q-evaluation by backward induction; pi(states) is the target's probability of measuring."""
    q_taken = np.zeros(len(df))
    v_state = np.zeros(len(df))
    next_cols = [f"next_{c}" for c in state_cols]
    quarter = df["quarter"].to_numpy()
    action = df["action"].to_numpy()
    models = None
    for t in range(n_quarters - 1, -1, -1):
        idx = np.flatnonzero(quarter == t)
        rows = df.iloc[idx]
        target = np.asarray(r, dtype=float)[idx].copy()
        if t < n_quarters - 1:
            nxt = rows[next_cols].set_axis(state_cols, axis=1)
            p_next = np.asarray(pi(nxt), dtype=float)
            target += gamma * (p_next * models[1].predict(nxt[state_cols]) + (1 - p_next) * models[0].predict(nxt[state_cols]))
        fitted = []
        for a in (0, 1):
            mask = action[idx] == a
            model = make_regressor()
            model.fit(rows.loc[mask, state_cols] if mask.any() else rows[state_cols],
                      target[mask] if mask.any() else np.zeros(len(rows)))
            fitted.append(model)
        models = tuple(fitted)
        q0, q1 = models[0].predict(rows[state_cols]), models[1].predict(rows[state_cols])
        p_now = np.asarray(pi(rows[state_cols]), dtype=float)
        q_taken[idx] = np.where(action[idx] == 1, q1, q0)
        v_state[idx] = p_now * q1 + (1 - p_now) * q0
    value = float(v_state[quarter == 0].mean())
    return FQEFit(value=value, q_taken=q_taken, v_state=v_state)


def main() -> None:
    import argparse

    from xgboost import XGBRegressor

    import run_pipeline as rp
    from research import simulator as sim
    from research import simulator_fqi as fqi
    from research import simulator_policy as pol
    from research import simulator_sweep as sweep
    from research.calibration import write_results

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patients", type=int, default=100000)
    parser.add_argument("--beta", type=float, nargs="+", default=sweep.BETAS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.perf_counter()
    source = sim.load_source()
    params = sim.fit_params(source)
    train_c = sweep.cohort_truth(source, params, fqi.TRAIN_PATIENTS, args.seed + 1)
    train_s, train_o = pol.static_features(train_c, source), pol.other_dx_months(train_c, source)
    eval_c = sweep.cohort_truth(source, params, args.patients, args.seed)
    eval_s, eval_o = pol.static_features(eval_c, source), pol.other_dx_months(eval_c, source)
    ids = eval_c.ages.index
    phase = pd.Series(mdp.guideline_phase(eval_c, args.seed), index=ids)
    reference = pd.read_csv(rp.TABLES_DIR / "simulator_true_values.csv")
    make_q = lambda: XGBRegressor(**XGB_REGRESSOR, random_state=args.seed)  # noqa: E731
    make_fqi = lambda: XGBRegressor(**fqi.XGB_REGRESSOR, random_state=args.seed)  # noqa: E731
    logged_dir = rp.PROJECT_ROOT / "data" / "processed" / "simulated" / "logged"
    print(f"cohorts ready | {time.perf_counter() - start:.0f}s")

    rows = []
    for beta in args.beta:
        df = pd.read_parquet(logged_dir / f"logged_beta_{beta:g}.parquet").sort_values(["patient_id", "quarter"])
        df = df.reset_index(drop=True)
        df["_pcode"] = pd.factorize(df["patient_id"], sort=True)[0]
        df["phase"] = df["next_phase"] = df["patient_id"].map(phase).to_numpy()
        base_cols = [c for c in pol.feature_columns(df) if f"next_{c}" in df.columns and c != "phase"]
        state_cols = base_cols + ["phase"]
        ref = reference[reference["beta"] == beta].set_index("schedule")

        train_logged = mdp.logged_dataset(train_c, source, params, beta, args.seed + 1, static=train_s, other=train_o)
        fqi_cols = [c for c in pol.feature_columns(train_logged) if f"next_{c}" in train_logged.columns]
        targets = {
            "never measure": (lambda s: np.zeros(len(s)), {lam: ref.loc["never measure", f"value_lambda_{lam:g}"]
                                                           for lam in EVAL_LAMBDAS}),
            "quarterly guideline": (lambda s: mdp.guideline_actions(s, s["phase"].to_numpy()).astype(float),
                                    {lam: ref.loc["quarterly guideline", f"value_lambda_{lam:g}"] for lam in EVAL_LAMBDAS}),
            "behavior (on-policy check)": (
                lambda s: mdp.FOLLOW_GUIDELINE * mdp.guideline_actions(s, s["phase"].to_numpy())
                + (1 - mdp.FOLLOW_GUIDELINE) * mdp.EXPLORE_RATE,
                {lam: ref.loc["behavior (guideline + 20% exploration)", f"value_lambda_{lam:g}"] for lam in EVAL_LAMBDAS}),
        }
        for lam in EVAL_LAMBDAS:
            models = fqi.fit_fqi(train_logged, lam, fqi_cols, make_fqi)
            greedy = fqi.greedy_policy(models, fqi_cols)
            run = pol.run_policy(eval_c, source, params, beta, args.seed, policy_fn=greedy, static=eval_s, other=eval_o)
            truth = mdp.true_values(run.detected, run.visits, ids, [lam])[lam]
            q0, q1 = fqi.q_values(models[0], train_logged.loc[train_logged["quarter"] == 0, fqi_cols])
            own = float(np.maximum(q0, q1).mean())

            def pi_fqi(s, models=models):
                quarter = int(s["month"].iloc[0]) // 3 if s["month"].nunique() == 1 else None
                if quarter is not None:
                    a0, a1 = fqi.q_values(models[quarter], s[fqi_cols])
                    return (a1 > a0).astype(float)
                out = np.zeros(len(s))
                for m, part in s.groupby("month"):
                    a0, a1 = fqi.q_values(models[int(m) // 3], part[fqi_cols])
                    out[s.index.get_indexer(part.index)] = (a1 > a0).astype(float)
                return out

            targets[f"FQI policy (lambda {lam:g})"] = (pi_fqi, {lam: truth}, own)

        for name, spec in targets.items():
            pi, truths = spec[0], spec[1]
            own = spec[2] if len(spec) > 2 else np.nan
            p_rows = np.asarray(pi(df[state_cols].reset_index(drop=True)), dtype=float)
            w = cumulative_weights(df, p_rows)
            w_prev = previous_weights(df, w)
            final_w = w[df["quarter"].to_numpy() == mdp.N_QUARTERS - 1]
            for lam, truth in truths.items():
                r = mdp_reward(df, lam)
                fit = fqe(df, r, pi, state_cols, make_q)
                est_is = wpdis(df, r, w)
                est_dr = wdr(df, r, w, fit.q_taken, fit.v_state, w_prev=w_prev)
                ci_is = bootstrap(lambda c: wpdis(df, r, w, patients=c), df, seed=args.seed)
                ci_dr = bootstrap(lambda c: wdr(df, r, w, fit.q_taken, fit.v_state, patients=c, w_prev=w_prev), df,
                                  seed=args.seed)
                rows.append({"beta": beta, "lambda": lam, "target": name, "true_value": truth,
                             "fqe": fit.value, "wpdis": est_is, "wpdis_lo": ci_is[0], "wpdis_hi": ci_is[1],
                             "wdr": est_dr, "wdr_lo": ci_dr[0], "wdr_hi": ci_dr[1],
                             "fqi_self_estimate": own if name.startswith("FQI") else np.nan,
                             "ess_final_quarter": ess(final_w), "matching_trajectories": int((final_w > 0).sum())})
                rr = rows[-1]
                print(f"beta={beta} lambda={lam:g} {name}: truth {truth:.0f} | FQE {rr['fqe']:.0f} | WPDIS {est_is:.0f} "
                      f"[{ci_is[0]:.0f}, {ci_is[1]:.0f}] | WDR {est_dr:.0f} [{ci_dr[0]:.0f}, {ci_dr[1]:.0f}] | "
                      f"ESS {rr['ess_final_quarter']:.0f} | {time.perf_counter() - start:.0f}s")
    table = pd.DataFrame(rows)
    for est in ["fqe", "wpdis", "wdr", "fqi_self_estimate"]:
        table[f"{est}_error"] = table[est] - table["true_value"]
        table[f"{est}_relative_error"] = table[f"{est}_error"] / table["true_value"].abs()
    table["wpdis_covers"] = (table["wpdis_lo"] <= table["true_value"]) & (table["true_value"] <= table["wpdis_hi"])
    table["wdr_covers"] = (table["wdr_lo"] <= table["true_value"]) & (table["true_value"] <= table["wdr_hi"])
    rp.save_metrics(table, rp.TABLES_DIR / "simulator_ope.csv")
    show = ["beta", "lambda", "target", "true_value", "fqe", "wpdis", "wdr", "fqi_self_estimate",
            "fqe_relative_error", "wpdis_relative_error", "wdr_relative_error", "ess_final_quarter"]
    with pd.option_context("display.width", 260, "display.max_rows", 200):
        print(table[show].round(3).to_string(index=False))
    log = table.assign(model="beta=" + table["beta"].map("{:g}".format) + " lambda=" + table["lambda"].map("{:g}".format)
                       + " " + table["target"], target="off-policy value estimate")
    write_results(log, task="21", seed=args.seed,
                  metrics=["true_value", "fqe", "wpdis", "wdr", "fqe_relative_error", "wpdis_relative_error",
                           "wdr_relative_error"],
                  notes=f"off-policy evaluation from {args.patients:,}-patient behavior-policy logs vs simulator truth; "
                        "FQE (XGBoost), weighted per-decision IS, weighted doubly robust")
    print(f"total {time.perf_counter() - start:.0f}s")


def mdp_reward(df: pd.DataFrame, lam: float) -> np.ndarray:
    from research.simulator_fqi import reward

    return reward(df, lam)


if __name__ == "__main__":
    main()
