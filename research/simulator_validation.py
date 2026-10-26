"""Simulator validation against the Kaggle data (ISEM 735 W6-2).

Every metric takes the five tables (Kaggle or simulated), so the same code scores both.
`validation_table` lines them up with a stated tolerance per target; results are written to
outputs/tables/simulator_validation.csv and discussed in research/simulator_validation.md.

    python -m research.simulator_validation --patients 100000
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research import simulator as sim
from src.data_cleaning import make_condition_events

CUTOFF = pd.Timestamp("2021-12-31")
AGE_BINS = [-np.inf, 49, 64, 79, np.inf]  # same bands as research/equity.py
AGE_LABELS = ["<50", "50-64", "65-79", "80+"]

# Absolute tolerances. Proportions are in [0, 1].
TOLERANCE = {
    "lab_visits_per_patient": 0.05,
    "lab_visits_per_patient_median": 0,
    "median_gap_days": 15,
    "charlson_spearman": 0.03,
    "progression_rate": 0.02,
    "prevalence": 0.02,
    "prevalence_by_age": 0.03,
    "incident_share": 0.05,
    "abnormal_rate": 0.02,
}


def lab_visit_metrics(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    visits = tables["lab_results"][["patient_id", "test_date"]].drop_duplicates()
    per_patient = visits.groupby("patient_id").size().reindex(tables["patients"]["patient_id"], fill_value=0)
    gaps = visits.sort_values(["patient_id", "test_date"]).groupby("patient_id")["test_date"].diff().dt.days
    rho = spearmanr(per_patient.to_numpy(),
                    tables["patients"].set_index("patient_id").loc[per_patient.index, "charlson_index"]).statistic
    return {"lab_visits_per_patient": per_patient.mean(), "lab_visits_per_patient_median": per_patient.median(),
            "median_gap_days": gaps.median(), "charlson_spearman": rho}


def progression_rate(tables: dict[str, pd.DataFrame]) -> float:
    """Share of the at-risk cohort (under 2 conditions by the cutoff) with 2 or more by 2024."""
    events = make_condition_events(tables["diagnoses"])
    ids = tables["patients"]["patient_id"]
    baseline = events[events["visit_date"] <= CUTOFF].groupby("patient_id")["condition"].nunique().reindex(ids, fill_value=0)
    future = events[events["visit_date"] <= sim.END_DATE].groupby("patient_id")["condition"].nunique().reindex(ids, fill_value=0)
    at_risk = baseline < 2
    return float((future[at_risk] >= 2).mean())


def condition_metrics(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per lab-linked condition: prevalence by 2024, by age band, and share first diagnosed after the cutoff."""
    events = make_condition_events(tables["diagnoses"])
    first = events[events["condition"].isin(sim.LAB_CONDITIONS)].groupby(["patient_id", "condition"])["visit_date"].min()
    patients = tables["patients"][["patient_id", "age"]].copy()
    patients["age_band"] = pd.cut(patients["age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    rows = []
    for condition in sim.LAB_CONDITIONS:
        dates = first.xs(condition, level="condition") if condition in first.index.get_level_values(1) else pd.Series(dtype="datetime64[ns]")
        has = patients["patient_id"].isin(dates.index)
        rows.append({"metric": "prevalence", "group": condition, "value": has.mean()})
        for band, part in patients.assign(has=has).groupby("age_band"):
            rows.append({"metric": "prevalence_by_age", "group": f"{condition} {band}", "value": part["has"].mean()})
        rows.append({"metric": "incident_share", "group": condition,
                     "value": float((dates > CUTOFF).mean()) if len(dates) else np.nan})
    return pd.DataFrame(rows)


def abnormal_rates(tables: dict[str, pd.DataFrame]) -> pd.Series:
    return tables["lab_results"].groupby("test_name")["is_abnormal"].mean()


def metric_frame(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [{"metric": k, "group": "", "value": v} for k, v in lab_visit_metrics(tables).items()]
    rows.append({"metric": "progression_rate", "group": "", "value": progression_rate(tables)})
    abnormal = abnormal_rates(tables)
    rows += [{"metric": "abnormal_rate", "group": t, "value": v} for t, v in abnormal.items()]
    return pd.concat([pd.DataFrame(rows), condition_metrics(tables)], ignore_index=True)


def validation_table(kaggle: pd.DataFrame, simulated: pd.DataFrame) -> pd.DataFrame:
    """Join two metric frames and mark each target within tolerance.

    Abnormal rates of the signal tests are reported but not scored: beta changes them by design,
    and the Kaggle labs track the all-time diagnosis flag rather than disease timing.
    """
    out = kaggle.merge(simulated, on=["metric", "group"], suffixes=("_kaggle", "_sim"), how="outer")
    out["tolerance"] = out["metric"].map(TOLERANCE)
    out["scored"] = ~((out["metric"] == "abnormal_rate") & out["group"].isin(sim.SIGNAL_TESTS))
    diff = (out["value_sim"] - out["value_kaggle"]).abs()
    out["passed"] = np.where(out["scored"], diff <= out["tolerance"] + 1e-12, np.nan)
    return out


def main() -> None:
    import argparse
    import time

    import run_pipeline as rp

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patients", type=int, default=100000)
    parser.add_argument("--beta", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.perf_counter()
    source = sim.load_source()
    params = sim.fit_params(source)
    kaggle = metric_frame(source)
    parts = []
    for beta in args.beta:
        tables, _ = sim.simulate_cohort(source, params, args.patients, beta, args.seed)
        parts.append(validation_table(kaggle, metric_frame(tables)).assign(beta=beta))
        print(f"beta={beta} done at {time.perf_counter() - start:.0f}s")
    table = pd.concat(parts, ignore_index=True)
    path = rp.TABLES_DIR / "simulator_validation.csv"
    rp.save_metrics(table, path)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table.round(4).to_string(index=False))
    scored = table[table["scored"]]
    print(f"\nPassed {int(scored['passed'].sum())} of {len(scored)} scored targets -> {path}")


if __name__ == "__main__":
    main()
