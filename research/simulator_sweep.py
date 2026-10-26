"""Beta sweep, part 1: non-adaptive schedules in the simulator (ISEM 735 W7-1, results-log task 16).

A schedule is a patient's full set of lab visits (it replaces background care; observed care,
the copied Kaggle visit dates, is one schedule among the others). Every visit returns the full
panel and costs research.cost_model.SIMULATOR_MEASUREMENT_COST. For each beta in {0, 0.25, 0.5,
1, 2} and each schedule, the sweep reports spend, detection delay from true onset (RQ3), the
share detected by labs, false-alarm visits, and age-band measurement shares (RQ4).

Schedules:
- observed care: the resampled patient's own Kaggle lab-visit dates
- fixed 3-, 6- and 12-month, from a random first month per patient
- random visits matched to the 12-month schedule's total (mean of 3 seeds)
- guideline: yearly once diabetes or kidney disease is known (ADA 2025, Section 11: eGFR and
  albuminuria at least annually); otherwise every 3 years from age 35 (ADA 2025, Section 2;
  USPSTF 2021), every 5 years from age 20 (ACC/AHA 2013: risk factors every 4 to 6 years), and
  no routine panel under 20. Heart failure has no guideline screening interval.

Truth (onsets, clinical dates) is the same as research.simulator.simulate_cohort's for the same
seed, so the beta sweep reuses the W6 cohorts' latent paths. Lab noise uses common random numbers
keyed by (patient, month, test): two schedules that measure a patient in the same month see the same
value, so differences between schedules come from timing, not noise. Prediction metrics (AUROC,
Brier) are the next task, scored against labels from true onsets.

    python -m research.simulator_sweep
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research import simulator as sim

BETAS = [0.0, 0.25, 0.5, 1.0, 2.0]
FIXED_INTERVALS = {"3-month": 3, "6-month": 6, "12-month": 12}
RANDOM_SEEDS = [1, 2, 3]
YEARS = sim.N_MONTHS / 12
TRIGGER_TESTS = list(sim.TRIGGERS)
GUIDELINE_DX = ("type2_diabetes", "chronic_kidney_disease")
AGE_BINS = [-np.inf, 49, 64, 79, np.inf]  # same bands as research/equity.py
AGE_LABELS = ["<50", "50-64", "65-79", "80+"]


@dataclass
class Cohort:
    truth: pd.DataFrame  # patient_id, condition, onset_month, clinical_date, clinical_month, ...
    ages: pd.Series  # age by patient_id
    observed: pd.DataFrame  # observed-care visits: patient_id, test_date
    mapping: pd.DataFrame  # patient_id, source_patient_id


def cohort_truth(source: dict[str, pd.DataFrame], params: sim.SimulatorParams, n_patients: int,
                 seed: int = 42) -> Cohort:
    """The latent truth simulate_cohort draws for the same seed, without labs or tables."""
    rng = np.random.default_rng(seed)
    source_ids = rng.choice(np.sort(source["patients"]["patient_id"].unique()), size=n_patients, replace=True)
    mapping = pd.DataFrame({"patient_id": [f"S{i + 1:07d}" for i in range(n_patients)],
                            "source_patient_id": source_ids})
    visits = mapping.merge(source["lab_results"][["patient_id", "test_date"]].drop_duplicates()
                           .rename(columns={"patient_id": "source_patient_id"}), on="source_patient_id")
    visits = visits[["patient_id", "test_date"]].sort_values(["patient_id", "test_date"]).reset_index(drop=True)
    truth = sim._draw_truth(mapping, source, params, visits, rng)
    truth["clinical_date"] = truth["clinical_date"].where(truth["clinical_month"] < sim.N_MONTHS)
    ages = mapping.merge(source["patients"][["patient_id", "age"]].rename(columns={"patient_id": "source_patient_id"}),
                         on="source_patient_id").set_index("patient_id")["age"]
    return Cohort(truth=truth, ages=ages, observed=visits, mapping=mapping)


_pnum = sim.patient_number
crn_uniform = sim.crn_uniform
crn_normal = sim.crn_normal


def _phase(patient_ids: pd.Series, interval: int, seed: int) -> np.ndarray:
    """First visit month in [0, interval). Intervals dividing 12 share one draw, so 3-, 6- and
    12-month schedules are nested (every 12-month visit is also a 6- and 3-month visit)."""
    base = 12 if 12 % interval == 0 else interval
    u = crn_uniform(_pnum(patient_ids), np.zeros(len(patient_ids), dtype=np.int64), np.zeros(len(patient_ids)),
                    seed, stream=2)
    return np.floor(u * base).astype(int) % interval


def fixed_schedule(patient_ids: pd.Series, interval: int, seed: int) -> pd.DataFrame:
    patient_ids = pd.Series(patient_ids).reset_index(drop=True)
    phase = _phase(patient_ids, interval, seed)
    reps = (sim.N_MONTHS - 1 - phase) // interval + 1
    idx = np.repeat(np.arange(len(patient_ids)), reps)
    step = np.arange(len(idx)) - np.repeat(np.cumsum(reps) - reps, reps)
    return pd.DataFrame({"patient_id": patient_ids.to_numpy()[idx], "month": phase[idx] + step * interval})


def random_schedule(patient_ids: pd.Series, n_measurements: int, seed: int) -> pd.DataFrame:
    patient_ids = pd.Series(patient_ids).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(patient_ids) * sim.N_MONTHS, size=n_measurements, replace=False)
    return pd.DataFrame({"patient_id": patient_ids.to_numpy()[cells // sim.N_MONTHS], "month": cells % sim.N_MONTHS})


def guideline_schedule(ages: pd.Series, dx_month: pd.DataFrame, phase: pd.Series) -> pd.DataFrame:
    """Visits under the guideline rule; the interval is set at each visit from what is known then."""
    known = (dx_month[dx_month["condition"].isin(GUIDELINE_DX)].dropna(subset=["diagnosis_month"])
             .groupby("patient_id")["diagnosis_month"].min().reindex(ages.index))
    base = np.where(ages >= 35, 36, np.where(ages >= 20, 60, 0))
    known_month = known.to_numpy()
    current = phase.reindex(ages.index).to_numpy().astype(float)
    current[(base == 0) & np.isnan(known_month)] = np.nan
    ids = ages.index.to_numpy()
    rows_id, rows_month = [], []
    while True:
        active = ~np.isnan(current) & (current < sim.N_MONTHS)
        if not active.any():
            break
        rows_id.append(ids[active])
        rows_month.append(current[active].astype(int))
        dx_known = ~np.isnan(known_month) & (known_month <= current)
        interval = np.where(dx_known, 12, base).astype(float)
        interval[interval == 0] = np.nan
        current = np.where(active, current + interval, np.nan)
    out = pd.DataFrame({"patient_id": np.concatenate(rows_id) if rows_id else [],
                        "month": np.concatenate(rows_month) if rows_month else []})
    return out.sort_values(["patient_id", "month"]).reset_index(drop=True)


def visits_from_months(months: pd.DataFrame) -> pd.DataFrame:
    """Month index -> visit on the 15th of that month."""
    m = months["month"].to_numpy().astype(int)
    dates = pd.to_datetime(pd.DataFrame({"year": 2018 + m // 12, "month": m % 12 + 1, "day": 15}))
    return pd.DataFrame({"patient_id": months["patient_id"].to_numpy(), "test_date": dates.to_numpy()})


def measure(visits: pd.DataFrame, truth: pd.DataFrame, params: sim.SimulatorParams, beta: float,
            seed: int) -> pd.DataFrame:
    """Trigger-test values at each visit, with common random numbers."""
    tests = [t for t in TRIGGER_TESTS if t in params.labs.index]
    labs = visits.loc[visits.index.repeat(len(tests)), ["patient_id", "test_date"]].reset_index(drop=True)
    labs["test_name"] = np.tile(tests, len(visits))
    labs["condition"] = labs["test_name"].map({t: c for t, (_, _, c) in sim.TRIGGERS.items()})
    labs = labs.merge(truth[["patient_id", "condition", "onset_month"]], on=["patient_id", "condition"], how="left")
    month = sim.month_index(labs["test_date"])
    window = labs["condition"].map(params.window).to_numpy()
    severity = sim.latent_severity(month, labs["onset_month"].to_numpy(), window)
    names = labs["test_name"].to_numpy()
    z = crn_normal(_pnum(labs["patient_id"]), month, sim.crn_test_index(names), seed)
    meta = params.labs.loc[names]
    values = sim.lab_means(params, names, severity, beta) + meta["healthy_sd"].to_numpy() * z
    labs["value"] = np.round(np.maximum(values, meta["floor"].to_numpy()), 3)
    labs["month"] = month
    return labs


def run_schedule(cohort: Cohort, visits: pd.DataFrame, params: sim.SimulatorParams, beta: float, seed: int,
                 return_labs: bool = False):
    labs = measure(visits, cohort.truth, params, beta, seed)
    detected = sim.detect(cohort.truth, labs)
    return (detected, labs) if return_labs else detected


def run_guideline(cohort: Cohort, params: sim.SimulatorParams, beta: float, seed: int, max_iter: int = 8):
    """Iterate schedule and detection until the schedule stops changing (it only ever grows)."""
    phase = pd.Series(_phase(pd.Series(cohort.ages.index), 12, seed), index=cohort.ages.index)
    detected = sim.detect(cohort.truth, cohort.observed.iloc[:0].assign(test_name=[], value=[]))
    previous = -1
    for _ in range(max_iter):
        months = guideline_schedule(cohort.ages, detected[["patient_id", "condition", "diagnosis_month"]], phase)
        visits = visits_from_months(months)
        detected, labs = run_schedule(cohort, visits, params, beta, seed, return_labs=True)
        if len(months) == previous:
            break
        previous = len(months)
    return detected, labs, visits


def delay_metrics(detected: pd.DataFrame) -> pd.DataFrame:
    """Per condition and overall, over onsets inside 2018-2024; undiagnosed onsets censored at month 84."""
    inside = detected[detected["onset_month"].between(0, sim.N_MONTHS - 1)].copy()
    inside["delay"] = inside["diagnosis_month"].fillna(sim.N_MONTHS).clip(upper=sim.N_MONTHS) - inside["onset_month"]
    rows = []
    for name, part in list(inside.groupby("condition")) + [("all", inside)]:
        rows.append({"condition": name, "onsets_in_window": len(part),
                     "restricted_mean_delay": part["delay"].mean(), "median_delay": part["delay"].median(),
                     "diagnosed_share": part["diagnosis_month"].notna().mean(),
                     "lab_detected_share": (part["route"] == "lab").mean()})
    return pd.DataFrame(rows)


def false_alarm_visits(labs: pd.DataFrame, detected: pd.DataFrame) -> int:
    """Visits where a trigger fired for a condition not yet begun (s < 1) and not yet diagnosed."""
    crossed = labs[sim.crosses_trigger(labs["test_name"].to_numpy(), labs["value"].to_numpy())]
    crossed = crossed.drop(columns="onset_month").merge(
        detected[["patient_id", "condition", "onset_month", "diagnosis_month"]], on=["patient_id", "condition"])
    early = crossed["onset_month"].isna() | (crossed["month"] < crossed["onset_month"])
    undiagnosed = crossed["diagnosis_month"].isna() | (crossed["month"] < crossed["diagnosis_month"])
    return int(crossed.loc[early & undiagnosed, ["patient_id", "test_date"]].drop_duplicates().shape[0])


def summarize(name: str, beta: float, cohort: Cohort, visits: pd.DataFrame, detected: pd.DataFrame,
              labs: pd.DataFrame) -> list[dict]:
    from research.cost_model import SIMULATOR_MEASUREMENT_COST

    n = len(cohort.ages)
    per_year = len(visits) / n / YEARS
    common = {"beta": beta, "schedule": name, "measurements_per_patient_year": per_year,
              "spend_per_patient_year": per_year * SIMULATOR_MEASUREMENT_COST,
              "false_alarm_visits_per_patient_year": false_alarm_visits(labs, detected) / n / YEARS}
    band = pd.cut(cohort.ages, AGE_BINS, labels=AGE_LABELS).astype(str)
    share = visits["patient_id"].map(band).value_counts(normalize=True) / band.value_counts(normalize=True)
    common |= {f"measurement_ratio_age_{b}": share.get(b, 0.0) for b in AGE_LABELS}
    rows = [common | r for r in delay_metrics(detected).to_dict("records")]
    banded = detected.assign(age_band=detected["patient_id"].map(band))
    for b, part in banded.groupby("age_band"):
        overall = delay_metrics(part).set_index("condition").loc["all"]
        rows[-1][f"restricted_mean_delay_age_{b}"] = overall["restricted_mean_delay"]
    return rows


def sweep(cohort: Cohort, params: sim.SimulatorParams, betas=BETAS, seed: int = 42, log=print) -> pd.DataFrame:
    ids = pd.Series(cohort.ages.index)
    schedules = {"observed care": cohort.observed}
    for name, interval in FIXED_INTERVALS.items():
        schedules[name] = visits_from_months(fixed_schedule(ids, interval, seed))
    budget = len(schedules["12-month"])
    rows = []
    start = time.perf_counter()
    for beta in betas:
        for name, visits in schedules.items():
            detected, labs = run_schedule(cohort, visits, params, beta, seed, return_labs=True)
            rows += summarize(name, beta, cohort, visits, detected, labs)
        parts = []
        for s in RANDOM_SEEDS:
            visits = visits_from_months(random_schedule(ids, budget, s))
            detected, labs = run_schedule(cohort, visits, params, beta, seed, return_labs=True)
            parts.append(pd.DataFrame(summarize("random (matched to 12-month)", beta, cohort, visits, detected, labs)))
        mean = pd.concat(parts).groupby(["beta", "schedule", "condition"], sort=False).mean(numeric_only=True)
        rows += mean.reset_index().to_dict("records")
        detected, labs, visits = run_guideline(cohort, params, beta, seed)
        rows += summarize("guideline", beta, cohort, visits, detected, labs)
        log(f"beta={beta} done | {time.perf_counter() - start:.0f}s")
    return pd.DataFrame(rows)


def main() -> None:
    import argparse

    import run_pipeline as rp
    from research.calibration import write_results

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patients", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.perf_counter()
    source = sim.load_source()
    params = sim.fit_params(source)
    cohort = cohort_truth(source, params, args.patients, args.seed)
    print(f"cohort ready in {time.perf_counter() - start:.0f}s")
    table = sweep(cohort, params, seed=args.seed)
    rp.save_metrics(table, rp.TABLES_DIR / "simulator_sweep_schedules.csv")
    show = ["beta", "schedule", "spend_per_patient_year", "restricted_mean_delay", "median_delay",
            "lab_detected_share", "diagnosed_share", "false_alarm_visits_per_patient_year"]
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table[table["condition"] == "all"][show].round(3).to_string(index=False))
    overall = table[table["condition"] == "all"].assign(target="detection (all lab-linked conditions)",
                                                        model=lambda d: "beta=" + d["beta"].map("{:g}".format)
                                                        + " " + d["schedule"])
    write_results(overall, task="16", seed=args.seed,
                  metrics=["spend_per_patient_year", "restricted_mean_delay", "lab_detected_share"],
                  notes=f"simulator schedule sweep, {args.patients:,} patients, non-adaptive schedules; "
                        "delay in months from true onset, censored at end of 2024")
    print(f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
