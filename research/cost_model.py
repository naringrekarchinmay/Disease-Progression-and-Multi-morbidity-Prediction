"""Measurement cost model (ISEM 735 proposal, weeks 2-3 deliverable).

One scheduled measurement is a lab panel plus an established-patient office visit.
Prices are 2026 Medicare national rates from CMS (research/cost_inputs.csv). Each of
the data's lab tests maps to the code that pays for it; a visit pays each code once,
plus one blood draw.

    python -m research.cost_model
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COST_INPUTS = Path(__file__).resolve().parent / "cost_inputs.csv"
PRICES = pd.read_csv(COST_INPUTS, dtype={"code": str}).set_index("code")["rate_usd"].to_dict()
BLOOD_DRAW, OFFICE_VISIT = "36415", "99213"

# Lab test in the data -> billing code that pays for it. eGFR is calculated from the
# creatinine result, so it has no code of its own.
TEST_TO_CODE: dict[str, str | None] = {
    "glucose_fasting": "80053", "creatinine": "80053", "potassium": "80053",
    "sodium": "80053", "ALT": "80053", "AST": "80053",
    "WBC": "85025", "hemoglobin": "85025",
    "LDL": "80061", "HDL": "80061", "total_cholesterol": "80061", "triglycerides": "80061",
    "HbA1c": "83036", "BNP": "83880", "troponin_I": "84484", "TSH": "84443",
    "eGFR": None,
}


def lab_visit_cost(test_names) -> float:
    """Cost of one lab visit: each billing code once, plus one blood draw."""
    codes = {TEST_TO_CODE[name] for name in test_names} - {None}
    return PRICES[BLOOD_DRAW] + sum(PRICES[code] for code in codes)


YEARS_OF_DATA = 7  # 2018-01-01 to 2024-12-31
SCHEDULES_PER_YEAR = {"observed care": None, "12-month": 1, "6-month": 2, "3-month": 4}


def visit_costs(labs: pd.DataFrame) -> pd.Series:
    """Lab cost of every (patient, date) lab visit in the data."""
    return labs.groupby(["patient_id", "test_date"])["test_name"].agg(lab_visit_cost)


def main() -> None:
    import run_pipeline as rp
    from research.calibration import write_results

    labs = pd.read_csv(rp.PROJECT_ROOT / "data" / "lab_results.csv", usecols=["patient_id", "test_date", "test_name"])
    costs = visit_costs(labs)
    lab_panel = float(costs.mean())
    measurement = lab_panel + PRICES[OFFICE_VISIT]
    visits_per_year = len(costs) / labs["patient_id"].nunique() / YEARS_OF_DATA

    summary = pd.DataFrame([{
        "lab_visits_priced": len(costs),
        "lab_panel_mean": lab_panel,
        "lab_panel_median": costs.median(),
        "lab_panel_p10": costs.quantile(0.1),
        "lab_panel_p90": costs.quantile(0.9),
        "office_visit": PRICES[OFFICE_VISIT],
        "measurement_cost": measurement,
        "office_visit_share": PRICES[OFFICE_VISIT] / measurement,
        "observed_measurements_per_patient_year": visits_per_year,
    }])
    schedules = pd.DataFrame([
        {"schedule": name,
         "measurements_per_year": visits_per_year if per_year is None else per_year,
         "annual_cost_per_patient": (visits_per_year if per_year is None else per_year) * measurement}
        for name, per_year in SCHEDULES_PER_YEAR.items()
    ])
    schedules["multiple_of_observed"] = schedules["annual_cost_per_patient"] / schedules["annual_cost_per_patient"].iloc[0]

    rp.save_metrics(summary, rp.TABLES_DIR / "cost_model_summary.csv")
    rp.save_metrics(schedules, rp.TABLES_DIR / "cost_model_schedules.csv")
    print(summary.T.round(4).to_string())
    print(schedules.round(3).to_string(index=False))

    log = summary.assign(target="cost model", model="lab panel + office visit (CMS 2026)")
    write_results(log, task="11", seed=rp.RANDOM_STATE,
                  metrics=["lab_panel_mean", "measurement_cost", "observed_measurements_per_patient_year"],
                  notes="CMS CLFS CY2026 Q4 and PFS RVU26D Oct 2026 national rates; research/cost_inputs.csv")


if __name__ == "__main__":
    main()
