# Lab lead time and the lab constraint: notes

Written 2026-11-09 (tasks 22 and 23). Code: `research/lead_time.py`. Tests: `tests/test_lead_time.py`.
Table: `outputs/tables/lab_lead_time.csv`. Prompted by Part 3 feedback, major issue 1: the lab artifact was
treated differently from the medication artifact.

## Task 22: how far ahead do Kaggle labs predict diagnosis?

**Method.** Incident cases are patients whose first diagnosis of a condition is in 2020 or later, so every
case has at least two years of history. Each of their labs drawn before that diagnosis is binned by years
before it and scored, by AUROC of the raw value, against all labs of the same test in patients never
diagnosed with the condition (eGFR scored with its sign reversed). A real biomarker should separate better
as the lab gets closer to diagnosis.

**Result: no lead-time gradient at all.**

| Test (condition) | 0-1 yr | 1-2 | 2-3 | 3-4 | 4-5 | 5-10 | Spearman, value vs years |
|---|---|---|---|---|---|---|---|
| Creatinine (kidney disease) | 0.982 | 0.982 | 0.982 | 0.983 | 0.983 | 0.982 | 0.014 |
| eGFR (kidney disease) | 0.719 | 0.717 | 0.713 | 0.716 | 0.725 | 0.712 | 0.004 |
| Fasting glucose (type 2 diabetes) | 0.726 | 0.716 | 0.723 | 0.724 | 0.718 | 0.731 | -0.003 |
| LDL (hyperlipidemia) | 0.620 | 0.626 | 0.624 | 0.620 | 0.622 | 0.618 | -0.003 |
| Total cholesterol (hyperlipidemia) | 0.627 | 0.626 | 0.628 | 0.629 | 0.627 | 0.620 | -0.002 |
| BNP (heart failure) | 0.490 | 0.507 | 0.472 | 0.503 | 0.463 | 0.550 | 0.007 |

Future kidney patients have median creatinine 5.5 to 5.9 mg/dL (kidney-failure range) up to ten years
before diagnosis; future diabetics have median fasting glucose about 170 mg/dL, above the 126 cutoff, five to
ten years before. In Whitehall II, glucose rose from about 104 to 133 mg/dL over the last three years before
diagnosis (Tabák et al., 2009). HbA1c shows AUROC 0.5 only because it is ordered almost exclusively for
diabetics, so its control pool is mostly type 1 diabetics; it is not evidence either way.

**Reading.** The generator writes the lifetime diagnosis into every lab value regardless of timing. That is
the same kind of artifact as the medication leak, and the October 12 decision to keep labs as they are
("labs showing a disease early is the premise of scheduling") was only valid if the lead time was
plausible. It is not.

## The lab constraint

`constrain_labs` replaces every linked lab (creatinine, eGFR, glucose, HbA1c, LDL, total cholesterol, BNP)
drawn more than the simulator's onset window W before the condition's first diagnosis (diabetes 36, kidney
48, heart failure 30, hyperlipidemia 24 months; Chinmay's choice, 2026-11-09, so the Kaggle arm and the
simulator agree) with a random draw from that test's values in never-diagnosed patients. Seed 42.
49,483 of 2,827,722 lab rows change. After it, AUROC outside W is 0.49 to 0.51 for every test; inside W it is
unchanged.

**Inside W the jump is still a step, not a rise**: a kidney patient's creatinine goes from normal to 5.7 at
W months before diagnosis. The constraint caps how early labs can warn; it does not make the magnitudes
realistic. State this in the report.

## Task 23: the second leak, and results with and without the artifact

**The first attempt backfired.** Applied to all 17 tests, the constraint *raised* the calibrated XGBoost's
AUROC from 0.8276 to 0.8365. Single lab features got weaker (creatinine 0.590 to 0.575, glucose 0.686 to
0.675 among patients without that condition at baseline), so the gain came from somewhere else: **which tests
were ordered**. Among at-risk patients without hyperlipidemia at baseline, those with any LDL test progress
at 0.546 against 0.264 overall, because LDL is ordered almost only for patients who are or will be
hyperlipidemic. After the constraint, "LDL ordered but normal" occurs almost only for future cases
(progression 0.618), a fingerprint the constraint itself created. HbA1c, eGFR, HDL, triglycerides and BNP
are ordered the same way.

**Fix (Chinmay's choice, 2026-11-09).** Two new lab versions, built by `research.lead_time.lab_version`:

- **panel**: keep only the eight tests drawn at every lab visit (glucose, creatinine, WBC, sodium, potassium,
  ALT, AST, hemoglobin; 257,664 rows each). Removes the ordering leak.
- **panel_w**: that panel with glucose and creatinine older than W replaced as above. Removes the lead-time
  artifact as well.

Code: `research/lab_constraint.py`; features in `data/processed/modeling_dataset_{panel,panel_w}.csv`;
frontiers in `outputs/tables/scheduling_frontier_masking_{panel,panel_w}.csv`. Same splits and seeds; the
check that patients, targets and every non-lab feature are identical passes for both.

| Labs | Calibrated XGBoost AUROC | Brier | Keep none | Labs' contribution (keep all - keep none) |
|---|---|---|---|---|
| All 17 tests (original) | 0.8276 | 0.1670 | 0.8172 | 0.0104 |
| Universal panel | 0.8267 | 0.1673 | 0.8168 | 0.0099 |
| Universal panel, constrained to W | 0.8284 | 0.1669 | 0.8175 | 0.0109 |

Masking frontier (Table 2) in the same three versions, AUROC:

| Schedule | Original | Panel | Panel + W |
|---|---|---|---|
| 12-month | 0.8281 | 0.8265 | 0.8282 |
| Threshold tau 0.2 / matched random | 0.8250 / 0.8229 | 0.8245 / 0.8221 | 0.8251 / 0.8234 |
| Threshold tau 0.3 / matched random | 0.8271 / 0.8261 | 0.8263 / 0.8250 | 0.8270 / 0.8263 |

**Reading.** Removing both artifacts leaves Table 2's conclusions where they were: labs add about 0.01 AUROC,
the threshold rule edges budget-matched random visits, and a fixed 12-month schedule does as well as keeping
every lab. The lead-time artifact matters little for this target because W (24 to 48 months) is about as
long as the three-year prediction horizon, so most pre-cutoff labs of future cases already fall inside W.
What the constraint cannot remove is the size of the jump inside W. Whether the small gaps between versions
are real is answered by the task 24 bootstrap (`research/ci_notes.md`).
