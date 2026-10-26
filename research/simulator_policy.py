"""Beta sweep, part 3: adaptive threshold policies (ISEM 735 W7-3, results-log task 18).

Every quarter (28 decision months over 2018-2024) the policy looks at what it knows about each patient
and decides whether to order a measurement (full panel, $214.49). What it knows: covariates, the
diagnoses made so far, and the latest and previous value of each trigger test with the months since
the last lab. A risk model turns that into the probability that a lab-linked condition has begun but
is not yet diagnosed, or will begin within 12 months. Two uncertainty rules, confirmed by Chinmay on
2026-10-26:

- closeness: measure when the calibrated risk is within tau of 50%;
- spread: measure when the standard deviation across 10 bootstrap-trained XGBoosts is above a
  threshold. This stands in for the GRU's Monte Carlo dropout SD (the proposal's signal), which would
  need PyTorch in the policy loop; both measure disagreement among plausible models.

The risk models are trained per beta on a separate cohort (seed + 1) measured by a random quarterly
policy at the 12-month schedule's rate. Thresholds are set per beta on a third cohort (seed + 2) so
spend matches the guideline schedule and the 12-month schedule, then the policies run on the seed-42
evaluation cohort used by tasks 16 and 17 (same truth and common random numbers), so every result is
comparable with the fixed schedules. Detection delay, false alarms and age shares come from
research.simulator_sweep; AUROC and Brier from research.simulator_prediction.

    python -m research.simulator_policy

This module loads XGBoost, so it must never import torch (see research/gru_model.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research import simulator as sim
from research import simulator_sweep as sweep

DECISION_MONTHS = np.arange(0, sim.N_MONTHS, 3)
HORIZON_MONTHS = 12
TESTS = list(sim.TRIGGERS)
N_BAGS = 10
TRAIN_PATIENTS = 30000
CALIBRATION_PATIENTS = 10000
TRAINING_RATE = 0.25  # one measurement a year on average, the 12-month schedule's rate
SMOKING = {"never": 0, "former": 1, "current": 2}
INSURANCE = {"commercial": 0, "medicaid": 1, "medicare": 2, "uninsured": 3}
XGB_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9, tree_method="hist",
                  eval_metric="logloss")


@dataclass
class PolicyRun:
    detected: pd.DataFrame
    labs: pd.DataFrame
    visits: pd.DataFrame
    lab_dx_month: pd.Series
    snapshots: pd.DataFrame | None = None


def static_features(cohort: sweep.Cohort, source: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = ["age", "sex", "bmi", "systolic_bp", "charlson_index", "smoking_status", "insurance_type"]
    pts = cohort.mapping.merge(source["patients"].reindex(columns=["patient_id"] + cols).rename(columns={"patient_id": "source_patient_id"}),
                               on="source_patient_id").set_index("patient_id").loc[cohort.ages.index]
    return pd.DataFrame({"age": pts["age"], "male": (pts["sex"] == "M").astype(int), "bmi": pts["bmi"],
                         "systolic_bp": pts["systolic_bp"], "charlson_index": pts["charlson_index"],
                         "smoking": pts["smoking_status"].map(SMOKING), "insurance": pts["insurance_type"].map(INSURANCE)})


def other_dx_months(cohort: sweep.Cohort, source: dict[str, pd.DataFrame]) -> np.ndarray:
    """First-diagnosis months of the 11 non-lab-linked conditions, padded with inf: (patients, k)."""
    from src.data_cleaning import make_condition_events

    events = make_condition_events(source["diagnoses"])
    events = events[~events["condition"].isin(sim.LAB_CONDITIONS)]
    first = events.groupby(["patient_id", "condition"])["visit_date"].min().reset_index()
    first["m"] = sim.month_index(first["visit_date"])
    first = cohort.mapping.merge(first.rename(columns={"patient_id": "source_patient_id"}), on="source_patient_id")
    index = {pid: i for i, pid in enumerate(cohort.ages.index)}
    k = max(int(first.groupby("patient_id").size().max()) if len(first) else 1, 1)
    out = np.full((len(index), k), np.inf)
    first["slot"] = first.groupby("patient_id").cumcount()
    out[first["patient_id"].map(index).to_numpy(), first["slot"].to_numpy()] = first["m"].to_numpy()
    return out


def random_score(seed: int):
    """Score for a random policy: measure when a uniform draw is above the threshold."""
    rng = np.random.default_rng(seed)
    return lambda features: rng.random(len(features))


def run_policy(cohort: sweep.Cohort, source: dict[str, pd.DataFrame], params: sim.SimulatorParams, beta: float,
               seed: int, score_fn, threshold: float, record: bool = False, static: pd.DataFrame | None = None,
               other: np.ndarray | None = None) -> PolicyRun:
    """Quarterly decisions: measure a patient when score_fn(features) >= threshold."""
    ids = cohort.ages.index.to_numpy()
    n = len(ids)
    index = pd.Series(np.arange(n), index=ids)
    static = static_features(cohort, source) if static is None else static
    other = other_dx_months(cohort, source) if other is None else other
    truth = cohort.truth.set_index(["patient_id", "condition"])
    onset = np.column_stack([truth["onset_month"].reindex(pd.MultiIndex.from_arrays([ids, [c] * n])).to_numpy()
                             for c in sim.LAB_CONDITIONS])
    clinical = np.column_stack([truth["clinical_month"].reindex(pd.MultiIndex.from_arrays([ids, [c] * n])).to_numpy()
                                for c in sim.LAB_CONDITIONS])
    clinical = np.nan_to_num(clinical, nan=np.inf)
    clinical_date = np.column_stack([
        truth["clinical_date"].reindex(pd.MultiIndex.from_arrays([ids, [c] * n])).fillna(pd.Timestamp.max).to_numpy()
        for c in sim.LAB_CONDITIONS])
    lab_dx = np.full((n, len(sim.LAB_CONDITIONS)), np.inf)
    last = np.full((n, len(TESTS)), np.nan)
    prev = np.full((n, len(TESTS)), np.nan)
    last_month = np.full(n, np.nan)
    n_labs = np.zeros(n, dtype=int)
    cond_index = {c: i for i, c in enumerate(sim.LAB_CONDITIONS)}
    test_index = {t: i for i, t in enumerate(TESTS)}
    lab_parts, snapshots = [], []

    for d in DECISION_MONTHS:
        known = np.minimum(lab_dx, np.where(clinical < d, clinical, np.inf)) < d
        features = pd.DataFrame(static.to_numpy(), columns=static.columns)
        features["month"] = d
        features["n_labs"] = n_labs
        features["months_since_lab"] = d - last_month
        for c, i in cond_index.items():
            features[f"known_{c}"] = known[:, i].astype(int)
        features["other_conditions"] = (other < d).sum(axis=1)
        for t, j in test_index.items():
            features[f"last_{t}"] = last[:, j]
            features[f"delta_{t}"] = last[:, j] - prev[:, j]
        if record:
            label = ((onset <= d + HORIZON_MONTHS) & ~known).any(axis=1)
            snapshots.append(features.assign(patient_id=ids, label=label.astype(int)))
        measure = np.asarray(score_fn(features)) >= threshold
        if not measure.any():
            continue
        visits = sweep.visits_from_months(pd.DataFrame({"patient_id": ids[measure], "month": d}))
        labs = sweep.measure(visits, cohort.truth, params, beta, seed)
        lab_parts.append(labs)
        rows = labs["patient_id"].map(index).to_numpy()
        cols = labs["test_name"].map(test_index).to_numpy()
        prev[rows, cols] = last[rows, cols]
        last[rows, cols] = labs["value"].to_numpy()
        last_month[measure] = d
        n_labs[measure] += 1
        # Diagnoses made at this visit (same rule as research.simulator.detect).
        hit = labs[sim.crosses_trigger(labs["test_name"].to_numpy(), labs["value"].to_numpy())]
        if len(hit):
            r = hit["patient_id"].map(index).to_numpy()
            c = hit["condition"].map(cond_index).to_numpy()
            ok = ((d >= onset[r, c]) & (hit["test_date"].to_numpy() < clinical_date[r, c])
                  & np.isinf(lab_dx[r, c]))
            lab_dx[r[ok], c[ok]] = d

    labs = (pd.concat(lab_parts, ignore_index=True) if lab_parts
            else sweep.measure(cohort.observed.iloc[:0], cohort.truth, params, beta, seed))
    visits = labs[["patient_id", "test_date"]].drop_duplicates().reset_index(drop=True)
    detected = sim.detect(cohort.truth, labs)
    lab_dx_series = pd.Series(lab_dx.ravel(), index=pd.MultiIndex.from_product([ids, sim.LAB_CONDITIONS]))
    lab_dx_series = lab_dx_series[np.isfinite(lab_dx_series)]
    snaps = pd.concat(snapshots, ignore_index=True) if record else None
    return PolicyRun(detected=detected, labs=labs, visits=visits, lab_dx_month=lab_dx_series, snapshots=snaps)


# --- risk models (XGBoost; never imported by the tests) -------------------------------------


def feature_columns(snapshots: pd.DataFrame) -> list[str]:
    return [c for c in snapshots.columns if c not in ("patient_id", "label")]


def train_risk_models(snapshots: pd.DataFrame, seed: int):
    """Calibrated XGBoost for the closeness rule, and N_BAGS bootstrap models for the spread rule."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from xgboost import XGBClassifier

    cols = feature_columns(snapshots)
    patients = np.sort(snapshots["patient_id"].unique())
    rng = np.random.default_rng(seed)
    val_ids = set(rng.choice(patients, size=len(patients) // 5, replace=False))
    is_val = snapshots["patient_id"].isin(val_ids).to_numpy()
    fit, val = snapshots[~is_val], snapshots[is_val]
    base = XGBClassifier(**XGB_PARAMS, random_state=seed).fit(fit[cols], fit["label"])
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic").fit(val[cols], val["label"])
    bags = []
    fit_ids = np.sort(fit["patient_id"].unique())
    groups = fit.groupby("patient_id").indices
    for b in range(N_BAGS):
        draw = rng.choice(fit_ids, size=len(fit_ids), replace=True)
        rows = np.concatenate([groups[p] for p in draw])
        part = fit.iloc[rows]
        bags.append(XGBClassifier(**XGB_PARAMS, random_state=seed + b).fit(part[cols], part["label"]))
    return cols, calibrated, bags


TIE_JITTER = 1e-4  # isotonic risks come in steps; a tiny per-decision jitter splits ties so spend can be matched


def _jitter(f: pd.DataFrame, seed: int) -> np.ndarray:
    return TIE_JITTER * sim.crn_uniform(np.arange(len(f)), f["month"].to_numpy(), np.zeros(len(f)), seed, stream=3)


def closeness_score(cols, calibrated, seed: int = 0):
    return lambda f: -np.abs(calibrated.predict_proba(f[cols])[:, 1] - 0.5) + _jitter(f, seed)


def spread_score(cols, bags):
    return lambda f: np.std([m.predict_proba(f[cols])[:, 1] for m in bags], axis=0)


def calibrate_threshold(run_fn, target_rate: float, lo: float, hi: float, iters: int = 20) -> tuple[float, float]:
    """Bisection on the threshold so the measured share of decision slots matches target_rate."""
    rate = np.nan
    for _ in range(iters):
        mid = (lo + hi) / 2
        rate = run_fn(mid)
        if rate > target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, rate


def main() -> None:
    import argparse

    import run_pipeline as rp
    from research import simulator_prediction as pred
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

    def prepared(n, seed):
        c = sweep.cohort_truth(source, params, n, seed)
        return c, static_features(c, source), other_dx_months(c, source)

    train_c, train_s, train_o = prepared(TRAIN_PATIENTS, args.seed + 1)
    cal_c, cal_s, cal_o = prepared(CALIBRATION_PATIENTS, args.seed + 2)
    eval_c, eval_s, eval_o = prepared(args.patients, args.seed)
    labels = pred.true_labels(eval_c, source)
    split = pred.split_ids(labels, args.seed)
    y_test = labels.set_index("patient_id").loc[split["test"], "label"].to_numpy()
    slots = len(DECISION_MONTHS)
    # Budgets from task 16 (full 2018-2024 spend per patient-year): guideline at this beta, and 12-month.
    reference = pd.read_csv(rp.TABLES_DIR / "simulator_sweep_schedules.csv")
    print(f"cohorts ready | {time.perf_counter() - start:.0f}s")

    rows = []
    for beta in args.beta:
        train = run_policy(train_c, source, params, beta, args.seed + 1, random_score(args.seed + 1),
                           1 - TRAINING_RATE, record=True, static=train_s, other=train_o)
        cols, calibrated, bags = train_risk_models(train.snapshots, args.seed)
        snaps = train.snapshots
        print(f"beta={beta}: trained on {len(snaps):,} snapshots (label rate {snaps['label'].mean():.3f}) | "
              f"{time.perf_counter() - start:.0f}s")
        ref = reference[(reference["beta"] == beta) & (reference["condition"] == "all")].set_index("schedule")
        budgets = {"guideline budget": ref.loc["guideline", "measurements_per_patient_year"],
                   "12-month budget": ref.loc["12-month", "measurements_per_patient_year"]}
        rules = {"closeness": (closeness_score(cols, calibrated, args.seed), -0.5, TIE_JITTER),
                 "spread": (spread_score(cols, bags), 0.0, 0.5)}
        for rule, (score, lo, hi) in rules.items():
            for budget, per_year in budgets.items():
                target = per_year * sweep.YEARS / slots

                def cal_rate(th):
                    run = run_policy(cal_c, source, params, beta, args.seed + 2, score, th, static=cal_s, other=cal_o)
                    return len(run.visits) / (len(cal_c.ages) * slots)

                threshold, cal = calibrate_threshold(cal_rate, target, lo, hi)
                run = run_policy(eval_c, source, params, beta, args.seed, score, threshold, static=eval_s, other=eval_o)
                name = f"{rule} rule, {budget}"
                summary = pd.DataFrame(sweep.summarize(name, beta, eval_c, run.visits, run.detected, run.labs))
                features = pred.schedule_features(source, params, eval_c, run.visits, beta, args.seed)
                risk = pred.fit_and_predict(features, labels, split, args.seed).to_numpy()
                pre = run.visits[run.visits["test_date"] <= pred.CUTOFF]
                pre_year = len(pre) / args.patients / 4
                prediction = {"pre_cutoff_spend_per_patient_year": pre_year * SIMULATOR_MEASUREMENT_COST,
                              **score_metrics(y_test, risk)}
                for r in summary.to_dict("records"):
                    rows.append(r | {"threshold": threshold, "target_measurements_per_patient_year": per_year}
                                | (prediction if r["condition"] == "all" else {}))
                overall = rows[-1]
                print(f"  {name}: spend ${overall['spend_per_patient_year']:.0f} (target ${per_year * SIMULATOR_MEASUREMENT_COST:.0f}) "
                      f"delay {overall['restricted_mean_delay']:.2f} AUROC {overall['roc_auc']:.4f} | "
                      f"{time.perf_counter() - start:.0f}s")
    table = pd.DataFrame(rows)
    rp.save_metrics(table, rp.TABLES_DIR / "simulator_sweep_policies.csv")
    show = ["beta", "schedule", "spend_per_patient_year", "restricted_mean_delay", "lab_detected_share",
            "false_alarm_visits_per_patient_year", "roc_auc", "brier", "measurement_ratio_age_<50",
            "measurement_ratio_age_80+"]
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(table[table["condition"] == "all"][show].round(4).to_string(index=False))
    overall = table[table["condition"] == "all"].assign(
        target="detection and true-onset progression",
        model=lambda d: "beta=" + d["beta"].map("{:g}".format) + " " + d["schedule"])
    write_results(overall, task="18", seed=args.seed,
                  metrics=["spend_per_patient_year", "restricted_mean_delay", "roc_auc", "brier"],
                  notes=f"adaptive quarterly policies, {args.patients:,} evaluation patients; closeness to 50% "
                        "and bagged-XGBoost spread; thresholds matched to guideline and 12-month spend")
    print(f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
