"""Semi-synthetic multimorbidity simulator, core (ISEM 735 W6-1).

Design and decisions D1-D4 are in research/simulator_spec.md. In short, each simulated patient
is a resampled Kaggle patient ("resample and overlay"):

- Covariates, admissions, background lab-visit dates and the diagnoses of the 11 conditions
  without a linked lab are copied, so those marginals match the data by construction.
- The four lab-linked conditions (type 2 diabetes, chronic kidney disease, heart failure,
  hyperlipidemia) get a hidden severity s(t) on a monthly grid over 2018-2024. If the source
  patient was diagnosed, that date is the clinical diagnosis and true onset (s = 1) is a lag
  before it; patients never diagnosed can still have an onset whose clinical diagnosis would
  fall after 2024, drawn from the data's incidence.
- Every lab visit returns the full panel (the data's test choice gives the diagnosis away).
  Lab value = healthy mean + beta x (diagnosed mean - healthy mean) x min(s, 1) + noise, with
  the healthy group's spread as noise. Beta = 0 makes labs uninformative.
- A value past the clinical cutoff diagnoses the condition only if s >= 1 (a false alarm
  costs the measurement and nothing else); otherwise it is diagnosed at the clinical date.
- Medications start at diagnosis. Hospital admissions are copied unchanged (D4).

Judgements, stated as such in the write-up (confirmed by Chinmay on 2026-10-26):
window W and clinical lag per condition (below); HbA1c healthy mean 5.4% with SD 0.5 and
HbA1c and BNP diagnosed means set to the cutoff plus one healthy SD, because the data shows no
contrast for either; hyperlipidemia's onset falls between the previous routine lab visit and
the clinical diagnosis ("diagnosed at the next routine lipid test").

numpy and pandas only. Run a 5,000-patient sample with:

    python -m research.simulator --patients 5000 --beta 1
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LAB_CONDITIONS = ("type2_diabetes", "chronic_kidney_disease", "heart_failure", "hyperlipidemia")
START_DATE = pd.Timestamp("2018-01-01")
END_DATE = pd.Timestamp("2024-12-31")
N_MONTHS = 84
INCIDENCE_BURN_IN_MONTHS = 12  # first diagnoses in 2018 are mostly prevalent, not incident

WINDOW_MONTHS = {"type2_diabetes": 36, "chronic_kidney_disease": 48, "heart_failure": 30, "hyperlipidemia": 24}
# Uniform integer draw per patient; None means "until the next routine lipid test".
CLINICAL_LAG_MONTHS = {"type2_diabetes": (24, 72), "chronic_kidney_disease": (24, 60),
                       "heart_failure": (6, 24), "hyperlipidemia": None}
ICD10 = {"type2_diabetes": "E11.9", "chronic_kidney_disease": "N18.3", "heart_failure": "I50.9",
         "hyperlipidemia": "E78.5"}
SPECIALTY = {"type2_diabetes": "endocrinology", "chronic_kidney_disease": "nephrology",
             "heart_failure": "cardiology", "hyperlipidemia": "internal_medicine"}

SIGNAL_TESTS = {"glucose_fasting": "type2_diabetes", "HbA1c": "type2_diabetes",
                "creatinine": "chronic_kidney_disease", "eGFR": "chronic_kidney_disease",
                "LDL": "hyperlipidemia", "total_cholesterol": "hyperlipidemia", "BNP": "heart_failure"}
# Clinical cutoffs: (operator, value, condition).
TRIGGERS = {"glucose_fasting": (">=", 126.0, "type2_diabetes"), "HbA1c": (">=", 6.5, "type2_diabetes"),
            "eGFR": ("<", 60.0, "chronic_kidney_disease"), "LDL": (">=", 160.0, "hyperlipidemia"),
            "BNP": (">", 100.0, "heart_failure")}
HBA1C_HEALTHY = (5.4, 0.5)
CUTOFF_ANCHORED = ("HbA1c", "BNP")


@dataclass
class SimulatorParams:
    """Everything fitted from the source data, plus the stated judgements."""

    labs: pd.DataFrame  # index test_name: unit, reference_low/high, healthy_mean/sd, diagnosed_mean, floor
    incidence: dict[str, float]  # monthly hazard of a first diagnosis
    med_rate: dict[str, float]  # share of diagnosed patients with a medication for the condition
    med_pool: pd.DataFrame  # medication rows for lab-linked indications, sampled when a source has none
    window: dict[str, int] = field(default_factory=lambda: dict(WINDOW_MONTHS))
    clinical_lag: dict[str, tuple[int, int] | None] = field(default_factory=lambda: dict(CLINICAL_LAG_MONTHS))


def month_index(dates: pd.Series) -> np.ndarray:
    """Months since January 2018 (January 2018 is month 0)."""
    dates = pd.to_datetime(pd.Series(dates))
    return ((dates.dt.year - START_DATE.year) * 12 + dates.dt.month - 1).to_numpy()


def latent_severity(months: np.ndarray, onset: np.ndarray, window: int | np.ndarray) -> np.ndarray:
    """Quadratic rise over the window before onset (after Tabak et al., 2009); 1 at onset, then above 1."""
    x = (np.asarray(months, dtype=float) - (np.asarray(onset, dtype=float) - window)) / window
    s = np.where(x > 0, x**2, 0.0)
    return np.nan_to_num(s, nan=0.0)


def lab_means(params: SimulatorParams, tests: np.ndarray, severity: np.ndarray, beta: float) -> np.ndarray:
    labs = params.labs.loc[tests]
    healthy = labs["healthy_mean"].to_numpy()
    diagnosed = labs["diagnosed_mean"].fillna(labs["healthy_mean"]).to_numpy()
    return healthy + beta * (diagnosed - healthy) * np.minimum(np.asarray(severity, dtype=float), 1.0)


def draw_lab_values(params: SimulatorParams, tests: np.ndarray, severity: np.ndarray, beta: float,
                    rng: np.random.Generator) -> np.ndarray:
    labs = params.labs.loc[tests]
    values = rng.normal(lab_means(params, tests, severity, beta), labs["healthy_sd"].to_numpy())
    return np.round(np.maximum(values, labs["floor"].to_numpy()), 3)


def lab_flags(params: SimulatorParams, tests: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    labs = params.labs.loc[tests]
    low, high = labs["reference_low"].to_numpy(), labs["reference_high"].to_numpy()
    flag = np.where(values < low, "low", np.where(values > high, "high", "normal"))
    return pd.DataFrame({"flag": flag, "is_abnormal": (flag != "normal").astype(int),
                         "delta_from_normal": np.round(values - (low + high) / 2, 3)})


def crosses_trigger(tests: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(tests), dtype=bool)
    for test, (op, cutoff, _) in TRIGGERS.items():
        mask = np.asarray(tests) == test
        v = np.asarray(values)[mask]
        out[mask] = {">=": v >= cutoff, ">": v > cutoff, "<": v < cutoff}[op]
    return out


def detect(truth: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    """Diagnosis date and route per (patient, condition).

    truth needs patient_id, condition, onset_month, clinical_date (NaT if after 2024);
    labs needs patient_id, test_date, test_name, value.
    """
    labs = labs[["patient_id", "test_date", "test_name", "value"]]
    hits = labs[crosses_trigger(labs["test_name"].to_numpy(), labs["value"].to_numpy())].copy()
    hits["condition"] = hits["test_name"].map({t: c for t, (_, _, c) in TRIGGERS.items()})
    hits = hits.merge(truth[["patient_id", "condition", "onset_month", "clinical_date"]],
                      on=["patient_id", "condition"])
    hits = hits[(month_index(hits["test_date"]) >= hits["onset_month"].to_numpy())
                & (hits["clinical_date"].isna() | (hits["test_date"] < hits["clinical_date"]))]
    lab_date = hits.groupby(["patient_id", "condition"])["test_date"].min().rename("lab_date")
    out = truth.drop(columns=["lab_date", "diagnosis_date", "route"], errors="ignore").merge(
        lab_date.reset_index(), on=["patient_id", "condition"], how="left")
    clinical = out["clinical_date"].where(out["clinical_date"] <= END_DATE)
    out["diagnosis_date"] = out["lab_date"].fillna(clinical)
    out["route"] = np.where(out["lab_date"].notna(), "lab", np.where(clinical.notna(), "clinical", "none"))
    out["diagnosis_month"] = np.where(out["diagnosis_date"].notna(),
                                      month_index(out["diagnosis_date"].fillna(START_DATE)), np.nan)
    return out


# --- fitting ----------------------------------------------------------------------------------


def _first_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    from src.data_cleaning import make_condition_events

    events = make_condition_events(diagnoses)
    return (events.groupby(["patient_id", "condition"])["visit_date"].min()
            .rename("first_dx").reset_index())


def fit_params(source: dict[str, pd.DataFrame]) -> SimulatorParams:
    labs, patients = source["lab_results"], source["patients"]
    rows = []
    for test, part in labs.groupby("test_name"):
        row = {"test_name": test, "unit": part["unit"].mode().iloc[0],
               "reference_low": part["reference_low"].median(), "reference_high": part["reference_high"].median(),
               "floor": part["value"].min(), "diagnosed_mean": np.nan}
        condition = SIGNAL_TESTS.get(test)
        if condition is not None and f"dx_{condition}" in patients.columns:
            ever = part["patient_id"].isin(patients.loc[patients[f"dx_{condition}"] == 1, "patient_id"])
            healthy = part.loc[~ever, "value"]
            row["healthy_mean"], row["healthy_sd"] = healthy.mean(), healthy.std(ddof=0)
            row["diagnosed_mean"] = part.loc[ever, "value"].mean()
        else:
            row["healthy_mean"], row["healthy_sd"] = part["value"].mean(), part["value"].std(ddof=0)
        rows.append(row)
    lab_table = pd.DataFrame(rows).set_index("test_name")
    if "HbA1c" in lab_table.index:
        lab_table.loc["HbA1c", ["healthy_mean", "healthy_sd"]] = HBA1C_HEALTHY
        lab_table.loc["HbA1c", "floor"] = min(lab_table.loc["HbA1c", "floor"], 4.0)
    for test in CUTOFF_ANCHORED:
        if test in lab_table.index:
            lab_table.loc[test, "diagnosed_mean"] = TRIGGERS[test][1] + lab_table.loc[test, "healthy_sd"]

    first = _first_diagnoses(source["diagnoses"])
    first["month"] = month_index(first["first_dx"])
    all_ids = patients["patient_id"].unique()
    incidence, med_rate = {}, {}
    meds = source["medications"]
    for condition in LAB_CONDITIONS:
        months = first.loc[first["condition"] == condition].set_index("patient_id")["month"]
        months = months.reindex(all_ids)
        at_risk = months.isna() | (months >= INCIDENCE_BURN_IN_MONTHS)
        exposure = (months[at_risk].fillna(N_MONTHS).clip(upper=N_MONTHS) - INCIDENCE_BURN_IN_MONTHS).sum()
        events = (months[at_risk] < N_MONTHS).sum()
        incidence[condition] = float(events / exposure) if exposure > 0 else 0.0
        diagnosed = months.dropna().index
        treated = meds.loc[meds["indication"] == condition, "patient_id"].unique()
        med_rate[condition] = float(np.isin(diagnosed, treated).mean()) if len(diagnosed) else 0.0
    med_pool = meds[meds["indication"].isin(LAB_CONDITIONS)].reset_index(drop=True)
    return SimulatorParams(labs=lab_table, incidence=incidence, med_rate=med_rate, med_pool=med_pool)


# --- cohort -----------------------------------------------------------------------------------


def _draw_truth(mapping: pd.DataFrame, source: dict[str, pd.DataFrame], params: SimulatorParams,
                visits: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Onset and clinical diagnosis per (simulated patient, lab-linked condition)."""
    first = _first_diagnoses(source["diagnoses"])
    first = first[first["condition"].isin(LAB_CONDITIONS)]
    truth = pd.MultiIndex.from_product([mapping["patient_id"], LAB_CONDITIONS],
                                       names=["patient_id", "condition"]).to_frame(index=False)
    truth = truth.merge(mapping, on="patient_id").merge(
        first.rename(columns={"patient_id": "source_patient_id", "first_dx": "clinical_date"}),
        on=["source_patient_id", "condition"], how="left")
    n = len(truth)
    anchored = truth["clinical_date"].notna().to_numpy()
    clinical_month = np.full(n, np.nan)
    clinical_month[anchored] = month_index(truth.loc[anchored, "clinical_date"])
    lag = np.full(n, np.nan)

    # Months since the previous background lab visit, for hyperlipidemia's "next routine lipid test".
    gap = np.full(n, np.nan)
    hl = anchored & (truth["condition"] == "hyperlipidemia").to_numpy()
    if hl.any():
        prior = truth.loc[hl, ["patient_id", "clinical_date"]].reset_index().merge(visits, on="patient_id")
        prior = prior[prior["test_date"] < prior["clinical_date"]]
        last = prior.groupby("index")["test_date"].max()
        gap_series = pd.Series(np.nan, index=truth.index[hl])
        gap_series.loc[last.index] = (month_index(truth.loc[last.index, "clinical_date"])
                                      - month_index(last.to_numpy()))
        gap[hl] = gap_series.to_numpy()

    for condition in LAB_CONDITIONS:
        is_c = (truth["condition"] == condition).to_numpy()
        bounds = params.clinical_lag[condition]
        lo, hi = bounds if bounds is not None else (1, params.window[condition])
        # Anchored: lag before the source's diagnosis.
        idx = np.flatnonzero(is_c & anchored)
        if bounds is None:
            upper = np.where(np.isnan(gap[idx]), hi, np.maximum(gap[idx], 1))
            lag[idx] = np.floor(rng.uniform(1, upper + 1))
        else:
            lag[idx] = rng.integers(lo, hi + 1, size=len(idx))
        # Never diagnosed in the data: maybe a clinical diagnosis in the hi months after 2024.
        idx = np.flatnonzero(is_c & ~anchored)
        p = min(1.0, params.incidence[condition] * hi)
        has = rng.random(len(idx)) < p
        idx = idx[has]
        clinical_month[idx] = rng.integers(N_MONTHS, N_MONTHS + hi, size=len(idx))
        lag[idx] = rng.integers(lo, hi + 1, size=len(idx))

    truth["clinical_month"] = clinical_month
    truth["lag_months"] = lag
    truth["onset_month"] = clinical_month - lag
    return truth


def _strip_linked(conditions: str, codes: str) -> tuple[list[str], list[str]]:
    names = [c.strip() for c in str(conditions).split("|") if c.strip()] if conditions else []
    icds = [c.strip() for c in str(codes).split("|")] if codes else []
    icds += [""] * (len(names) - len(icds))
    keep = [(n, i) for n, i in zip(names, icds) if n not in LAB_CONDITIONS]
    return [n for n, _ in keep], [i for _, i in keep]


def _rewrite_diagnoses(copied: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Remove lab-linked conditions, then add them back from the simulated diagnosis date on."""
    dx = truth.dropna(subset=["diagnosis_date"])
    dx_dates = {pid: dict(zip(g["condition"], g["diagnosis_date"])) for pid, g in dx.groupby("patient_id")}
    rows = []
    for row in copied.to_dict("records"):
        names, icds = _strip_linked(row["secondary_diagnoses"], row["secondary_icd10s"])
        primary, primary_icd = row["primary_diagnosis"], row["primary_icd10"]
        if primary in LAB_CONDITIONS:
            if not names:
                continue
            primary, primary_icd = names.pop(0), icds.pop(0)
        for condition, date in dx_dates.get(row["patient_id"], {}).items():
            if row["visit_date"] > date:
                names.append(condition)
                icds.append(ICD10[condition])
        rows.append(row | {"primary_diagnosis": primary, "primary_icd10": primary_icd,
                           "secondary_diagnoses": "|".join(names), "secondary_icd10s": "|".join(icds)})
    added = pd.DataFrame({
        "patient_id": dx["patient_id"].to_numpy(), "visit_date": dx["diagnosis_date"].to_numpy(),
        "visit_type": "outpatient", "primary_diagnosis": dx["condition"].to_numpy(),
        "primary_icd10": dx["condition"].map(ICD10).to_numpy(), "secondary_diagnoses": "",
        "secondary_icd10s": "", "provider_specialty": dx["condition"].map(SPECIALTY).to_numpy(),
    })
    out = pd.concat([pd.DataFrame(rows, columns=copied.columns), added[copied.columns]], ignore_index=True)
    return out.sort_values(["patient_id", "visit_date"], kind="stable").reset_index(drop=True)


def _shift_to(template: pd.DataFrame, patient_id: str, date: pd.Timestamp) -> pd.DataFrame:
    shifted = template.copy()
    shifted["start_date"] = date + (shifted["start_date"] - shifted["start_date"].min())
    shifted["patient_id"] = patient_id
    return shifted


def _build_medications(source_meds: pd.DataFrame, mapping: pd.DataFrame, diagnoses: pd.DataFrame,
                       truth: pd.DataFrame, params: SimulatorParams, rng: np.random.Generator) -> pd.DataFrame:
    columns = source_meds.columns
    copied = mapping.merge(source_meds.rename(columns={"patient_id": "source_patient_id"}), on="source_patient_id")
    # Non-linked indications: start no earlier than the copied first diagnosis (as research/data.py does).
    other = copied[~copied["indication"].isin(LAB_CONDITIONS)]
    first = _first_diagnoses(diagnoses).rename(columns={"condition": "indication"})
    other = other.merge(first, on=["patient_id", "indication"], how="inner")
    other["start_date"] = other[["start_date", "first_dx"]].max(axis=1)
    parts = [other[columns]]

    linked = copied[copied["indication"].isin(LAB_CONDITIONS)]
    by_source = {key: g for key, g in linked.groupby(["patient_id", "indication"])}
    pools = {c: [g for _, g in params.med_pool[params.med_pool["indication"] == c].groupby("patient_id")]
             for c in LAB_CONDITIONS}
    for row in truth.dropna(subset=["diagnosis_date"]).itertuples(index=False):
        template = by_source.get((row.patient_id, row.condition))
        if template is None and pools[row.condition] and rng.random() < params.med_rate[row.condition]:
            template = pools[row.condition][rng.integers(len(pools[row.condition]))]
        if template is not None:
            parts.append(_shift_to(template[columns], row.patient_id, row.diagnosis_date))
    meds = pd.concat(parts, ignore_index=True)
    meds = meds[meds["start_date"] <= END_DATE]
    return meds.sort_values(["patient_id", "start_date", "medication"], kind="stable").reset_index(drop=True)


def simulate_cohort(source: dict[str, pd.DataFrame], params: SimulatorParams, n_patients: int, beta: float,
                    seed: int = 42) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Five Kaggle-schema tables for n_patients simulated patients, and the ground-truth table.

    source is the cleaned Kaggle data (src.data_cleaning.clean_all), not the leak-free medications:
    medication starts are rebuilt here.
    """
    rng = np.random.default_rng(seed)
    source_ids = rng.choice(np.sort(source["patients"]["patient_id"].unique()), size=n_patients, replace=True)
    mapping = pd.DataFrame({"patient_id": [f"S{i + 1:07d}" for i in range(n_patients)],
                            "source_patient_id": source_ids})

    def copy(table: pd.DataFrame) -> pd.DataFrame:
        out = mapping.merge(table.rename(columns={"patient_id": "source_patient_id"}), on="source_patient_id")
        return out.drop(columns="source_patient_id")[table.columns]

    lab_cols = source["lab_results"].columns
    visits = copy(source["lab_results"][["patient_id", "test_date"]].drop_duplicates())
    visits = visits.sort_values(["patient_id", "test_date"]).reset_index(drop=True)

    truth = _draw_truth(mapping, source, params, visits, rng)
    truth["clinical_date"] = truth["clinical_date"].where(truth["clinical_month"] < N_MONTHS)

    tests = params.labs.index.to_numpy()
    labs = visits.loc[visits.index.repeat(len(tests))].reset_index(drop=True)
    labs["test_name"] = np.tile(tests, len(visits))
    labs["condition"] = labs["test_name"].map(SIGNAL_TESTS)
    labs = labs.merge(truth[["patient_id", "condition", "onset_month"]], on=["patient_id", "condition"], how="left")
    window = labs["condition"].map(params.window).fillna(1).to_numpy()
    severity = latent_severity(month_index(labs["test_date"]), labs["onset_month"].to_numpy(), window)
    labs["value"] = draw_lab_values(params, tests=labs["test_name"].to_numpy(), severity=severity, beta=beta, rng=rng)
    meta = params.labs.loc[labs["test_name"], ["unit", "reference_low", "reference_high"]].reset_index(drop=True)
    labs = pd.concat([labs, meta, lab_flags(params, labs["test_name"].to_numpy(), labs["value"].to_numpy())], axis=1)

    truth = detect(truth, labs)

    patients = copy(source["patients"])
    diagnosed = truth.assign(flag=truth["diagnosis_date"].notna().astype(int)).pivot(
        index="patient_id", columns="condition", values="flag")
    for condition in LAB_CONDITIONS:
        if f"dx_{condition}" in patients.columns:
            patients[f"dx_{condition}"] = patients["patient_id"].map(diagnosed[condition]).astype(int)

    diagnoses = _rewrite_diagnoses(copy(source["diagnoses"]), truth)
    tables = {
        "patients": patients,
        "diagnoses": diagnoses,
        "lab_results": labs[lab_cols].sort_values(["patient_id", "test_date"], kind="stable").reset_index(drop=True),
        "medications": _build_medications(source["medications"], mapping, diagnoses, truth, params, rng),
        "outcomes": copy(source["outcomes"]),
    }
    truth = truth[["patient_id", "source_patient_id", "condition", "onset_month", "clinical_month", "lag_months",
                   "clinical_date", "diagnosis_date", "diagnosis_month", "route"]]
    return tables, truth.sort_values(["patient_id", "condition"]).reset_index(drop=True)


def load_source() -> dict[str, pd.DataFrame]:
    from src.data_cleaning import clean_all, load_raw_data

    return clean_all(load_raw_data())


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--patients", type=int, default=5000)
    parser.add_argument("--beta", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.perf_counter()
    source = load_source()
    params = fit_params(source)
    print(f"Loaded and fitted in {time.perf_counter() - start:.1f}s")
    print(params.labs.round(3).to_string())
    print("monthly incidence:", {k: round(v, 5) for k, v in params.incidence.items()})
    print("medication rate:", {k: round(v, 3) for k, v in params.med_rate.items()})
    for beta in args.beta:
        t0 = time.perf_counter()
        tables, truth = simulate_cohort(source, params, args.patients, beta, args.seed)
        diagnosed = truth.dropna(subset=["diagnosis_date"])
        delay = diagnosed["diagnosis_month"] - diagnosed["onset_month"]
        print(f"\nbeta={beta}: {time.perf_counter() - t0:.1f}s | rows " +
              ", ".join(f"{k} {len(v):,}" for k, v in tables.items()))
        summary = truth.groupby(["condition", "route"]).size().unstack(fill_value=0)
        summary["median_delay_months"] = delay.groupby(diagnosed["condition"]).median()
        print(summary.to_string())


if __name__ == "__main__":
    main()
