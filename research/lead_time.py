"""How far before diagnosis do labs separate future cases? (ISEM 735, task 22)

Part 3 feedback asked whether the lab artifact is the same kind of generator flaw as the
medication leak. A real biomarker rises toward diagnosis, so a lab drawn closer to the
diagnosis date should separate future cases from never-diagnosed patients better than one
drawn years earlier. Here each pre-diagnosis lab of an incident case (first diagnosis in
2020 or later, so at least two years of history) is binned by years before diagnosis and
scored against every lab of patients never diagnosed with that condition.

`constrain_labs` builds the comparison data for task 23: a linked lab drawn earlier than the
simulator's onset window W before diagnosis is replaced by a value drawn from the
never-diagnosed patients' values for the same test, so labs may show a disease only as far
ahead as the simulator lets them. On its own it leaves a second leak (which tests were ordered),
so task 23 applies it to the universal panel only (`lab_version`).

    python -m research.lead_time

No XGBoost or torch here.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import run_pipeline as rp
from research.simulator import WINDOW_MONTHS
from src.data_cleaning import make_condition_events

INCIDENT_FROM = pd.Timestamp("2020-01-01")
YEAR_BINS = [0, 1, 2, 3, 4, 5, 10]
MIN_LABS_PER_BIN = 20
# Lab -> the condition it observes. HbA1c is scored but not read as evidence: it is ordered
# almost only for diagnosed diabetics, so its never-diagnosed pool is mostly type 1 diabetics.
LINKED_TESTS = {"creatinine": "chronic_kidney_disease", "eGFR": "chronic_kidney_disease",
                "glucose_fasting": "type2_diabetes", "HbA1c": "type2_diabetes",
                "LDL": "hyperlipidemia", "total_cholesterol": "hyperlipidemia", "BNP": "heart_failure"}
# The eight tests drawn at every Kaggle lab visit (257,664 rows each). Every other test is ordered
# mostly for patients who are or will be diagnosed, so whether it was ordered leaks the diagnosis.
UNIVERSAL_PANEL = ("glucose_fasting", "creatinine", "WBC", "sodium", "potassium", "ALT", "AST", "hemoglobin")
LAB_VERSIONS = ("original", "panel", "panel_w")
# Sign that makes "more abnormal" score higher (eGFR falls with kidney disease).
DIRECTION = {"eGFR": -1}
DAYS_PER_MONTH = 365.25 / 12


def first_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """One row per (patient_id, condition) with its first diagnosis date as `dx_date`."""
    events = make_condition_events(diagnoses)
    return (events.groupby(["patient_id", "condition"])["visit_date"].min()
            .rename("dx_date").reset_index())


def lead_time_table(labs: pd.DataFrame, first: pd.DataFrame, tests=LINKED_TESTS,
                    bins=YEAR_BINS, incident_from=INCIDENT_FROM) -> pd.DataFrame:
    """AUROC of each linked lab for 'future case' against 'never diagnosed', by years before diagnosis."""
    rows = []
    for test, condition in tests.items():
        dx = first[first["condition"] == condition]
        t = labs.loc[labs["test_name"] == test, ["patient_id", "test_date", "value"]]
        controls = t.loc[~t["patient_id"].isin(set(dx["patient_id"])), "value"].to_numpy()
        cases = t.merge(dx[dx["dx_date"] >= incident_from], on="patient_id")
        cases = cases[cases["test_date"] < cases["dx_date"]]
        years = (cases["dx_date"] - cases["test_date"]).dt.days / 365.25
        rho = spearmanr(years, cases["value"]).correlation if len(cases) > 10 else np.nan
        sign = DIRECTION.get(test, 1)
        for lo, hi in zip(bins[:-1], bins[1:]):
            values = cases.loc[(years >= lo) & (years < hi), "value"].to_numpy()
            if len(values) < MIN_LABS_PER_BIN or len(controls) == 0:
                continue
            y = np.r_[np.ones(len(values)), np.zeros(len(controls))]
            rows.append({
                "test": test, "condition": condition, "years_before": f"{lo}-{hi}",
                "n_case_labs": len(values), "n_case_patients": cases.loc[(years >= lo) & (years < hi),
                                                                         "patient_id"].nunique(),
                "case_median": float(np.median(values)), "control_median": float(np.median(controls)),
                "auroc": float(roc_auc_score(y, sign * np.r_[values, controls])),
                "spearman_value_vs_years": float(rho),
            })
    return pd.DataFrame(rows)


def constrain_labs(labs: pd.DataFrame, first: pd.DataFrame, window_months=WINDOW_MONTHS,
                   tests=LINKED_TESTS, seed: int = rp.RANDOM_STATE) -> pd.DataFrame:
    """Replace linked labs drawn more than W months before their condition's diagnosis.

    The replacement is a random draw (with replacement) from the same test's values in
    patients never diagnosed with that condition. Labs of never-diagnosed patients, labs
    within W of diagnosis and labs after it are unchanged; so are dates, units and which
    tests were ordered. Flags are recomputed from the reference range for replaced rows.
    """
    rng = np.random.default_rng(seed)
    out = labs.copy()
    replaced = np.zeros(len(out), dtype=bool)
    for test, condition in tests.items():
        dx = first.loc[first["condition"] == condition, ["patient_id", "dx_date"]]
        is_test = (out["test_name"] == test).to_numpy()
        dx_date = out["patient_id"].map(dx.set_index("patient_id")["dx_date"])
        controls = out.loc[is_test & dx_date.isna().to_numpy(), "value"].to_numpy()
        limit = dx_date - pd.to_timedelta(window_months[condition] * DAYS_PER_MONTH, unit="D")
        hit = is_test & (out["test_date"] < limit).to_numpy()
        if hit.any() and len(controls):
            out.loc[hit, "value"] = rng.choice(controls, size=int(hit.sum()), replace=True)
            replaced |= hit
    if replaced.any() and {"reference_low", "reference_high"} <= set(out.columns):
        part = out.loc[replaced]
        flag = np.where(part["value"] < part["reference_low"], "low",
                        np.where(part["value"] > part["reference_high"], "high", "normal"))
        if "flag" in out.columns:
            out.loc[replaced, "flag"] = flag
        if "is_abnormal" in out.columns:
            out.loc[replaced, "is_abnormal"] = (flag != "normal").astype(int)
        if "delta_from_normal" in out.columns:
            out.loc[replaced, "delta_from_normal"] = (
                part["value"] - (part["reference_low"] + part["reference_high"]) / 2).round(3)
    out.attrs["replaced"] = int(replaced.sum())
    return out


def lab_version(labs: pd.DataFrame, first: pd.DataFrame, version: str, seed: int = rp.RANDOM_STATE) -> pd.DataFrame:
    """'original': all labs; 'panel': only the universal panel; 'panel_w': the panel with linked labs
    older than W replaced (constrain_labs)."""
    if version == "original":
        return labs
    panel = labs[labs["test_name"].isin(UNIVERSAL_PANEL)].reset_index(drop=True)
    if version == "panel":
        return panel
    if version == "panel_w":
        return constrain_labs(panel, first, seed=seed)
    raise ValueError(f"unknown lab version {version!r}; expected one of {LAB_VERSIONS}")


def main() -> None:
    from research.calibration import write_results
    from research.data import load_clean_data

    start = time.perf_counter()
    data = load_clean_data()
    first = first_diagnoses(data["diagnoses"])
    labs = data["lab_results"]
    original = lead_time_table(labs, first).assign(labs="original")
    constrained_labs = constrain_labs(labs, first)
    constrained = lead_time_table(constrained_labs, first).assign(labs="constrained to window W")
    panel_w = lab_version(labs, first, "panel_w")
    panel = lead_time_table(panel_w, first).assign(labs="universal panel, constrained to W")
    table = pd.concat([original, constrained, panel], ignore_index=True)
    rp.save_metrics(table, rp.TABLES_DIR / "lab_lead_time.csv")
    show = ["labs", "test", "years_before", "n_case_labs", "case_median", "control_median", "auroc",
            "spearman_value_vs_years"]
    print(table[show].round(3).to_string(index=False))
    print(f"constrained: {constrained_labs.attrs['replaced']:,} of {len(labs):,} lab rows replaced")

    log = table.assign(target="lead time", model=table["labs"] + " " + table["test"] + " "
                       + table["years_before"] + " yr")
    write_results(log, task="22", seed=rp.RANDOM_STATE, metrics=["auroc"],
                  notes="lab value AUROC, incident cases (first dx 2020+) vs never-diagnosed, by years before dx")
    print(f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
