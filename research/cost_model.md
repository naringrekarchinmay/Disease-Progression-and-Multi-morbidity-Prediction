# Measurement cost model

Written 2026-09-28. Code: `research/cost_model.py`. Prices: `research/cost_inputs.csv`.
Tables: `outputs/tables/cost_model_summary.csv`, `outputs/tables/cost_model_schedules.csv`.

## What one measurement costs

A scheduled measurement is a lab panel plus an established-patient office visit. In the data,
lab visits almost never fall on the same day as a diagnosis visit (0.1%), so a scheduled draw is
treated as its own encounter, and chronic-care follow-up usually pairs results with a visit.

Prices are 2026 Medicare national rates, taken from two CMS files downloaded 2026-09-28:

- Clinical Laboratory Fee Schedule, CY2026 Q4 (file 26CLABQ4, 2,243 codes).
- Physician Fee Schedule relative value file RVU26D (October 2026 release). The office visit
  (99213) is 2.85 non-facility RVUs times the non-QP conversion factor of $33.4009, or $95.19.

Each lab test in the data maps to the code that pays for it. A visit pays each code once, plus
one blood draw ($9.34). The metabolic panel ($10.56) covers glucose, sodium, potassium,
creatinine, ALT and AST; eGFR is calculated from creatinine and costs nothing extra. The blood
count ($7.77) covers WBC and hemoglobin. The four lipid tests are one lipid panel ($13.39).
HbA1c ($9.71), BNP ($39.26), troponin ($12.47) and TSH ($16.80) are billed separately.

| Quantity | Value |
|---|---|
| Lab visits priced | 257,537 |
| Lab panel, mean (median; 10th to 90th percentile) | $44.57 ($41.06; $27.67 to $92.79) |
| Office visit | $95.19 |
| One measurement | $139.76 |
| Office visit share of a measurement | 68% |

Every visit includes the draw, blood count and metabolic panel ($27.67). The spread above that
comes from the lipid panel (59% of visits) and the less common add-ons, mainly BNP.

## Reference schedules

The data average 0.37 lab visits per patient per year (2.58 over seven years).

| Schedule | Measurements per year | Annual cost per patient | Multiple of observed |
|---|---|---|---|
| Observed care | 0.37 | $51 | 1.0 |
| 12-month | 1 | $140 | 2.7 |
| 6-month | 2 | $280 | 5.4 |
| 3-month | 4 | $559 | 10.9 |

These anchor the budget sweep in weeks 7 and 8: a controller's spending can be read as a
multiple of what the data's care costs, or against a fixed schedule.

## Limits

- Medicare national rates, not commercial prices or local adjustments (no geographic index).
- The office visit is priced at low complexity (99213); a moderate visit (99214) would raise it.
- Patient time, travel and the value of reassurance are not costed. The proposal notes this as
  a limitation of any prediction-driven schedule.
- The lab mix per measurement is the data's average mix. A controller that chose which tests to
  order would need per-test prices, which the code already supports through `lab_visit_cost`.
