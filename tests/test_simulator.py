"""Tests for the semi-synthetic simulator core (research/simulator.py, ISEM 735 W6-1).

Run with `pytest tests/`. The source cohort here is a tiny hand-built Kaggle-like dataset,
so every test runs in well under a second and never reads data/*.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import simulator as sim
from src.data_cleaning import clean_all, make_condition_events

TESTS = ["glucose_fasting", "HbA1c", "creatinine", "eGFR", "LDL", "BNP", "WBC"]
KAGGLE_COLUMNS = {
    "patients": ["patient_id", "age", "sex", "insurance_type", "charlson_index", "dx_type2_diabetes",
                 "dx_hyperlipidemia", "dx_heart_failure", "dx_chronic_kidney_disease", "dx_asthma"],
    "diagnoses": ["patient_id", "visit_date", "visit_type", "primary_diagnosis", "primary_icd10",
                  "secondary_diagnoses", "secondary_icd10s", "provider_specialty"],
    "lab_results": ["patient_id", "test_date", "test_name", "value", "unit", "reference_low",
                    "reference_high", "flag", "is_abnormal", "delta_from_normal"],
    "medications": ["patient_id", "medication", "dose", "unit", "frequency", "indication", "start_date",
                    "duration_days", "is_generic", "adherence_pct"],
    "outcomes": ["patient_id", "admission_date", "discharge_date", "length_of_stay_days"],
}


def _labs(pid: str, dates: list[str]) -> list[dict]:
    rows = []
    for date in dates:
        for test in TESTS:
            rows.append({"patient_id": pid, "test_date": date, "test_name": test, "value": 1.0, "unit": "u",
                         "reference_low": 0.0, "reference_high": 2.0, "flag": "normal", "is_abnormal": 0,
                         "delta_from_normal": 0.0})
    return rows


@pytest.fixture()
def source() -> dict[str, pd.DataFrame]:
    """P1 diabetic (dx 2021-06-15) with asthma, P2 never diagnosed, P3 kidney disease (dx 2020-03-01)."""
    patients = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3"], "age": [60, 45, 75], "sex": ["F", "M", "F"],
        "insurance_type": ["commercial", "medicaid", "medicare"], "charlson_index": [2, 0, 3],
        "dx_type2_diabetes": [1, 0, 0], "dx_hyperlipidemia": [0, 0, 0], "dx_heart_failure": [0, 0, 0],
        "dx_chronic_kidney_disease": [0, 0, 1], "dx_asthma": [1, 0, 0],
    })
    diagnoses = pd.DataFrame([
        {"patient_id": "P1", "visit_date": "2019-02-01", "visit_type": "outpatient", "primary_diagnosis": "asthma",
         "primary_icd10": "J45.909", "secondary_diagnoses": "", "secondary_icd10s": "",
         "provider_specialty": "pulmonology"},
        {"patient_id": "P1", "visit_date": "2021-06-15", "visit_type": "outpatient",
         "primary_diagnosis": "type2_diabetes", "primary_icd10": "E11.9", "secondary_diagnoses": "asthma",
         "secondary_icd10s": "J45.909", "provider_specialty": "endocrinology"},
        {"patient_id": "P1", "visit_date": "2023-01-10", "visit_type": "telehealth", "primary_diagnosis": "asthma",
         "primary_icd10": "J45.909", "secondary_diagnoses": "type2_diabetes", "secondary_icd10s": "E11.9",
         "provider_specialty": "internal_medicine"},
        {"patient_id": "P3", "visit_date": "2020-03-01", "visit_type": "outpatient",
         "primary_diagnosis": "chronic_kidney_disease", "primary_icd10": "N18.3", "secondary_diagnoses": "",
         "secondary_icd10s": "", "provider_specialty": "nephrology"},
    ])
    labs = pd.DataFrame(_labs("P1", ["2019-03-01", "2021-01-10", "2023-05-05"]) + _labs("P2", ["2020-02-02"])
                        + _labs("P3", ["2018-05-05", "2022-07-07"]))
    medications = pd.DataFrame([
        {"patient_id": "P1", "medication": "Metformin", "dose": 500.0, "unit": "mg", "frequency": "daily",
         "indication": "type2_diabetes", "start_date": "2019-01-01", "duration_days": 30, "is_generic": 1,
         "adherence_pct": 80.0},
        {"patient_id": "P1", "medication": "Albuterol", "dose": 90.0, "unit": "mcg", "frequency": "prn",
         "indication": "asthma", "start_date": "2018-06-01", "duration_days": 30, "is_generic": 1,
         "adherence_pct": 70.0},
        {"patient_id": "P3", "medication": "Lisinopril", "dose": 10.0, "unit": "mg", "frequency": "daily",
         "indication": "chronic_kidney_disease", "start_date": "2019-09-01", "duration_days": 30, "is_generic": 1,
         "adherence_pct": 90.0},
    ])
    outcomes = pd.DataFrame({"patient_id": ["P3"], "admission_date": ["2022-05-06"],
                             "discharge_date": ["2022-05-08"], "length_of_stay_days": [2]})
    return clean_all({"patients": patients, "diagnoses": diagnoses, "lab_results": labs,
                      "medications": medications, "outcomes": outcomes})


@pytest.fixture()
def params() -> sim.SimulatorParams:
    labs = pd.DataFrame({
        "test_name": TESTS,
        "unit": ["mg/dL", "%", "mg/dL", "mL/min/1.73m2", "mg/dL", "pg/mL", "K/uL"],
        "reference_low": [70.0, 4.0, 0.6, 60.0, 0.0, 0.0, 4.5],
        "reference_high": [99.0, 5.6, 1.2, 120.0, 99.0, 100.0, 11.0],
        "healthy_mean": [100.0, 5.4, 0.9, 90.0, 64.0, 50.0, 7.0],
        "healthy_sd": [10.0, 0.5, 0.1, 10.0, 20.0, 16.0, 1.0],
        "diagnosed_mean": [170.0, 7.0, 5.7, 35.0, 142.0, 117.0, np.nan],
        "floor": [40.0, 3.0, 0.3, 5.0, 5.0, 0.0, 1.0],
    }).set_index("test_name")
    pool = pd.DataFrame([
        {"patient_id": "X9", "medication": "Insulin", "dose": 10.0, "unit": "units", "frequency": "daily",
         "indication": "type2_diabetes", "start_date": pd.Timestamp("2020-01-01"), "duration_days": 30,
         "is_generic": 0, "adherence_pct": 75.0},
    ])
    return sim.SimulatorParams(
        labs=labs,
        incidence={c: 0.0 for c in sim.LAB_CONDITIONS},
        med_rate={c: 1.0 for c in sim.LAB_CONDITIONS},
        med_pool=pool,
    )


# --- latent path --------------------------------------------------------------------------


def test_latent_severity_is_zero_before_the_window_one_at_onset_and_keeps_rising():
    onset, window = 50, 36
    months = np.array([0, 13, 14, 32, 50, 60])
    s = sim.latent_severity(months, np.full(6, onset), window)
    assert s[0] == 0.0 and s[1] == 0.0 and s[2] == 0.0      # window starts at month 14
    assert s[3] == pytest.approx(0.25)                       # halfway through the window, quadratic
    assert s[4] == pytest.approx(1.0)
    assert s[5] > 1.0
    assert np.all(np.diff(s) >= 0)


def test_latent_severity_is_zero_without_an_onset():
    s = sim.latent_severity(np.array([0, 40, 83]), np.full(3, np.nan), 24)
    assert np.all(s == 0.0)


# --- lab mapping ---------------------------------------------------------------------------


def test_lab_means_ignore_severity_at_beta_zero(params):
    tests = np.array(["glucose_fasting", "eGFR", "LDL"])
    for s in [0.0, 0.5, 1.0, 3.0]:
        means = sim.lab_means(params, tests, np.full(3, s), beta=0.0)
        assert means == pytest.approx([100.0, 90.0, 64.0])


def test_lab_means_reach_the_diagnosed_mean_at_beta_one_and_cap_severity(params):
    tests = np.array(["glucose_fasting", "eGFR", "BNP", "WBC"])
    full = sim.lab_means(params, tests, np.array([1.0, 1.0, 1.0, 1.0]), beta=1.0)
    assert full == pytest.approx([170.0, 35.0, 117.0, 7.0])   # WBC carries no signal
    over = sim.lab_means(params, tests, np.array([4.0, 4.0, 4.0, 4.0]), beta=1.0)
    assert over == pytest.approx(full)
    half = sim.lab_means(params, tests, np.array([0.5, 0.5, 0.5, 0.5]), beta=2.0)
    assert half == pytest.approx([170.0, 35.0, 117.0, 7.0])


def test_lab_flags_follow_the_reference_range(params):
    tests = np.array(["glucose_fasting", "glucose_fasting", "glucose_fasting"])
    flags = sim.lab_flags(params, tests, np.array([60.0, 80.0, 150.0]))
    assert flags["flag"].tolist() == ["low", "normal", "high"]
    assert flags["is_abnormal"].tolist() == [1, 0, 1]
    assert flags["delta_from_normal"].tolist() == pytest.approx([60 - 84.5, 80 - 84.5, 150 - 84.5])


def test_triggers_use_the_clinical_cutoffs():
    tests = np.array(["glucose_fasting", "glucose_fasting", "HbA1c", "eGFR", "eGFR", "LDL", "BNP", "WBC"])
    values = np.array([126.0, 125.9, 6.5, 59.9, 60.0, 160.0, 100.5, 99.0])
    assert sim.crosses_trigger(tests, values).tolist() == [True, False, True, True, False, True, True, False]


# --- detection -----------------------------------------------------------------------------


def test_detection_needs_both_a_crossing_and_true_onset():
    truth = pd.DataFrame({"patient_id": ["A", "B", "C"], "condition": ["type2_diabetes"] * 3,
                          "onset_month": [30, 30, 30],
                          "clinical_date": pd.to_datetime(["2022-06-01", "2022-06-01", "2022-06-01"])})
    labs = pd.DataFrame({
        "patient_id": ["A", "A", "B", "C"],
        "test_date": pd.to_datetime(["2020-01-05", "2020-09-05", "2020-01-05", "2023-01-05"]),
        "test_name": ["glucose_fasting"] * 4,
        "value": [200.0, 200.0, 110.0, 200.0],
    })
    # A: month 24 crossing is before onset (false alarm), month 32 crossing after onset -> lab dx.
    # B: never crosses -> clinical. C: crossing only after the clinical date -> clinical.
    out = sim.detect(truth, labs).set_index("patient_id")
    assert out.loc["A", "route"] == "lab"
    assert out.loc["A", "diagnosis_date"] == pd.Timestamp("2020-09-05")
    assert out.loc["B", "route"] == "clinical"
    assert out.loc["B", "diagnosis_date"] == pd.Timestamp("2022-06-01")
    assert out.loc["C", "route"] == "clinical"


def test_detection_leaves_conditions_with_a_post_2024_clinical_date_undiagnosed():
    truth = pd.DataFrame({"patient_id": ["A"], "condition": ["chronic_kidney_disease"], "onset_month": [90],
                          "clinical_date": [pd.NaT]})
    labs = pd.DataFrame({"patient_id": ["A"], "test_date": pd.to_datetime(["2024-06-01"]),
                         "test_name": ["eGFR"], "value": [30.0]})
    out = sim.detect(truth, labs).iloc[0]
    assert out["route"] == "none" and pd.isna(out["diagnosis_date"])


# --- full cohort ---------------------------------------------------------------------------


def test_cohort_has_the_kaggle_schema_and_runs_through_the_cleaning_code(source, params):
    tables, truth = sim.simulate_cohort(source, params, n_patients=20, beta=1.0, seed=1)
    for name, columns in KAGGLE_COLUMNS.items():
        assert list(tables[name].columns) == columns, name
    clean_all(tables)
    make_condition_events(tables["diagnoses"])
    assert set(truth.columns) >= {"patient_id", "source_patient_id", "condition", "onset_month",
                                  "clinical_month", "diagnosis_date", "route"}
    assert len(truth) == 20 * len(sim.LAB_CONDITIONS)


def test_cohort_is_reproducible_by_seed(source, params):
    a = sim.simulate_cohort(source, params, n_patients=30, beta=1.0, seed=7)
    b = sim.simulate_cohort(source, params, n_patients=30, beta=1.0, seed=7)
    c = sim.simulate_cohort(source, params, n_patients=30, beta=1.0, seed=8)
    for name in a[0]:
        pd.testing.assert_frame_equal(a[0][name], b[0][name])
    pd.testing.assert_frame_equal(a[1], b[1])
    assert not a[0]["lab_results"]["value"].equals(c[0]["lab_results"]["value"])


def test_every_lab_visit_returns_the_full_panel(source, params):
    tables, _ = sim.simulate_cohort(source, params, n_patients=20, beta=0.5, seed=3)
    per_visit = tables["lab_results"].groupby(["patient_id", "test_date"])["test_name"].apply(set)
    assert all(tests == set(TESTS) for tests in per_visit)


def test_lab_linked_diagnoses_appear_only_from_the_simulated_diagnosis_date(source, params):
    tables, truth = sim.simulate_cohort(source, params, n_patients=40, beta=1.0, seed=5)
    events = make_condition_events(tables["diagnoses"])
    linked = events[events["condition"].isin(sim.LAB_CONDITIONS)]
    first = linked.groupby(["patient_id", "condition"])["visit_date"].min()
    expected = truth.dropna(subset=["diagnosis_date"]).set_index(["patient_id", "condition"])["diagnosis_date"]
    pd.testing.assert_series_equal(first.sort_index(), expected.sort_index(), check_names=False)
    # Non-linked conditions are copied unchanged (P1's asthma on 2019-02-01).
    asthma = events[events["condition"] == "asthma"].groupby("patient_id")["visit_date"].min()
    assert (asthma == pd.Timestamp("2019-02-01")).all()


def test_patient_dx_flags_match_simulated_diagnoses(source, params):
    tables, truth = sim.simulate_cohort(source, params, n_patients=40, beta=1.0, seed=5)
    patients = tables["patients"].set_index("patient_id")
    for row in truth.itertuples():
        assert patients.loc[row.patient_id, f"dx_{row.condition}"] == int(pd.notna(row.diagnosis_date))


def test_medications_start_on_or_after_diagnosis(source, params):
    tables, truth = sim.simulate_cohort(source, params, n_patients=40, beta=1.0, seed=5)
    meds = tables["medications"]
    first_dx = (make_condition_events(tables["diagnoses"]).groupby(["patient_id", "condition"])["visit_date"]
                .min().rename("first_dx").reset_index().rename(columns={"condition": "indication"}))
    merged = meds.merge(first_dx, on=["patient_id", "indication"], how="left")
    assert merged["first_dx"].notna().all()                  # no medication without its diagnosis
    assert (merged["start_date"] >= merged["first_dx"]).all()
    linked = merged[merged["indication"].isin(sim.LAB_CONDITIONS)]
    starts = linked.groupby(["patient_id", "indication"])["start_date"].min()
    dx = linked.groupby(["patient_id", "indication"])["first_dx"].min()
    assert (starts == dx).all()                              # lab-linked drugs start at diagnosis
    assert (meds["start_date"] <= sim.END_DATE).all()


def test_anchored_onset_precedes_the_source_diagnosis_by_the_clinical_lag(source, params):
    _, truth = sim.simulate_cohort(source, params, n_patients=60, beta=0.0, seed=11)
    p1 = truth[(truth["source_patient_id"] == "P1") & (truth["condition"] == "type2_diabetes")]
    assert len(p1) > 0
    clinical_month = sim.month_index(pd.Series([pd.Timestamp("2021-06-15")]))[0]
    lag = clinical_month - p1["onset_month"]
    lo, hi = sim.CLINICAL_LAG_MONTHS["type2_diabetes"]
    assert lag.between(lo, hi).all()
    assert (p1["clinical_month"] == clinical_month).all()


def test_beta_zero_labs_carry_no_information_about_severity(params):
    rng = np.random.default_rng(0)
    n = 20000
    tests = np.array(["glucose_fasting"] * n)
    severity = rng.uniform(0, 2, n)
    values = sim.draw_lab_values(params, tests, severity, beta=0.0, rng=rng)
    assert abs(np.corrcoef(severity, values)[0, 1]) < 0.03
    values_1 = sim.draw_lab_values(params, tests, severity, beta=1.0, rng=rng)
    assert np.corrcoef(severity, values_1)[0, 1] > 0.5


def test_fit_params_reads_lab_stats_and_overrides(source):
    fitted = sim.fit_params(source)
    assert set(fitted.labs.index) == set(TESTS)
    # HbA1c and BNP use the cutoff anchor: cutoff plus one healthy SD.
    assert fitted.labs.loc["HbA1c", "healthy_mean"] == pytest.approx(5.4)
    assert fitted.labs.loc["HbA1c", "diagnosed_mean"] == pytest.approx(6.5 + fitted.labs.loc["HbA1c", "healthy_sd"])
    assert fitted.labs.loc["BNP", "diagnosed_mean"] == pytest.approx(100 + fitted.labs.loc["BNP", "healthy_sd"])
    assert np.isnan(fitted.labs.loc["WBC", "diagnosed_mean"])
    assert set(fitted.incidence) == set(sim.LAB_CONDITIONS)
