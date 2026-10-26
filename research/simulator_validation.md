# Simulator validation (W6-2)

Written 2026-10-26. Code: `research/simulator_validation.py`. Tests: `tests/test_simulator_validation.py`.
Table: `outputs/tables/simulator_validation.csv` (100,000 simulated patients, seed 42, beta 0 and 1).

## Method

The same metric code scores the Kaggle data and a simulated cohort. Validation targets are checked at
beta = 0, the setting where labs carry no information about the hidden severity; beta = 1 is shown to
make clear what the signal changes. The tests repeat the beta = 0 check on 20,000 patients, so the
tolerances also absorb sampling noise.

| Target | Tolerance (absolute) |
|---|---|
| Lab visits per patient (mean; median) | 0.05; exact |
| Median gap between lab visits | 15 days |
| Spearman correlation of lab visits with Charlson index | 0.03 |
| Multimorbidity progression in the at-risk cohort | 2 percentage points |
| Prevalence by 2024, per lab-linked condition | 2 points |
| Prevalence by condition and age band | 3 points |
| Share of diagnoses first made after 2021 | 5 points |
| Abnormal rate, tests without a signal | 2 points |

## Results at beta = 0 (100,000 patients)

38 of 39 scored targets pass.

| Target | Kaggle | Simulated | Pass |
|---|---|---|---|
| Lab visits per patient | 2.575 (median 2) | 2.575 (median 2) | yes |
| Median gap | 431 days | 429 days | yes |
| Lab visits vs Charlson (Spearman) | -0.000 | -0.002 | yes |
| Progression, at-risk cohort | 43.5% | 44.4% | yes |
| Prevalence: kidney disease | 18.6% | 18.8% | yes |
| Prevalence: heart failure | 2.4% | 2.4% | yes |
| Prevalence: hyperlipidemia | 44.4% | 44.3% | yes |
| Prevalence: type 2 diabetes | 13.0% | 15.2% | **no** |
| Incident share (four conditions) | 21% to 27% | within 0.6 points | yes |
| Prevalence by age band (16 cells) | | all within 3 points; diabetes cells 1.9 to 2.4 points high | yes |
| Abnormal rate, 10 tests without a signal | 0.1% to 0.3% (triglycerides 100%) | within 0.1 point | yes |

The lab-visit targets match almost exactly because visit dates are copied from the resampled patients.
The progression rate matching is a real check: diagnoses of the four lab-linked conditions are rebuilt,
and the rebuild does not distort who progresses.

**The diabetes miss.** Of 15,203 simulated diabetes diagnoses at beta = 0, 2,099 belong to patients never
diagnosed in the data whose onset was drawn late in the window (clinical diagnosis after 2024) and who were
then caught by a routine glucose test. That is the whole 2.2-point excess. It follows from the W6-1
decision to trigger on the clinical cutoff of 126 mg/dL: Kaggle's never-diabetic glucose averages 131 mg/dL,
so 45% of tests after onset trigger a correct diagnosis whatever beta is. The test marks it as an expected
failure rather than widening the tolerance. Another 4,362 anchored diabetics are lab-detected earlier than
their clinical date at beta = 0, which moves dates but not prevalence.

**Abnormal rates of the signal tests are not scored.** Kaggle's labs follow the all-time diagnosis flag,
so they are abnormal throughout the record (eGFR 91%, LDL 94%, HbA1c 100%), mostly because those tests are
ordered almost only for diagnosed patients. The simulator draws every test at every visit and raises
values only near and after onset, so these rates cannot and should not match. At beta = 0 they sit at the
healthy-spread rate (eGFR 0.3%), except glucose (85%, against the 70-99 reference range).

## What beta = 1 changes

| Condition | Diagnosed, beta 0 (lab-detected) | Diagnosed, beta 1 (lab-detected) | Median delay from onset, beta 0 to 1 |
|---|---|---|---|
| Kidney disease | 18,817 (23) | 22,075 (12,082) | 42 to 31 months |
| Type 2 diabetes | 15,203 (6,461) | 16,212 (9,652) | 38 to 32 |
| Hyperlipidemia | 44,316 (33) | 44,716 (1,941) | 8 to 8 |
| Heart failure | 2,364 (0) | 2,437 (671) | 15 to 12 |

Prevalence of kidney disease rises 3.4 points and the progression rate to 46.5%, because labs catch onsets
whose clinical diagnosis would have come after 2024. That is the intended effect, not a calibration error.
Hyperlipidemia barely moves under background care: its onset falls after the previous routine visit by
construction, so only extra measurements can catch it early.

## Measurement cost in the simulator

Every simulated lab visit returns all 17 tests, so a simulated measurement costs the full panel: draw
$9.34, blood count $7.77, metabolic panel $10.56, lipid panel $13.39, HbA1c $9.71, BNP $39.26, troponin
$12.47 and TSH $16.80, for $119.30, plus the $95.19 office visit, or **$214.49**
(`research.cost_model.SIMULATOR_MEASUREMENT_COST`). Fixed schedules then cost $214 (12-month), $429
(6-month) and $858 (3-month) per patient-year, and the data's 0.37 visits per year cost $79.

## Not checked here

- That labs add nothing to the predictor at beta = 0 and clear signal at beta = 1 (the masking frontier):
  that needs the feature pipeline and XGBoost, which is W6-3.
- Medication and admission marginals. Admissions are copied; lab-linked medications are rebuilt at
  diagnosis, and every diagnosed Kaggle patient has one, so the rate is 1.0.

## W6-3: end-to-end smoke run (task 15)

Code: `research/simulator_smoke.py`. Tables: `outputs/tables/simulator_smoke.csv` and
`simulator_smoke_checks.csv`. Cohorts (100,000 patients, seed 42) are saved under
`data/processed/simulated/beta_0/` and `beta_1/` (gitignored) for weeks 7-8.

For each beta the unchanged leak-free feature pipeline (`src/features.py`) ran on the simulated tables,
the calibrated XGBoost was refit with and without the lab features (`lab_*` columns), and masking
schedules were scored on the test split with the all-features model, as `research/scheduling.py` does
on the Kaggle data. The keep-all schedule reproduces the direct test score exactly (asserted).

| | beta 0 | beta 1 |
|---|---|---|
| Progression rate, at-risk cohort | 44.4% | 46.5% |
| AUROC, all features | 0.8053 | 0.7877 |
| AUROC, no lab features | 0.8099 | 0.7809 |
| Masking: keep all / 6-month / 12-month / random / keep none | 0.8053 / 0.8054 / 0.8049 / 0.8053 / 0.8067 | 0.7877 / 0.7872 / 0.7869 / 0.7848 / 0.7723 |
| Calibration slope, all features | 0.95 | 0.97 |

Checks (tolerances are judgements for a test split of about 10,000 at-risk patients (a quarter of the cohort, as on the Kaggle data), where AUROC has a
standard error near 0.005):

| Check | Value | Rule | Pass |
|---|---|---|---|
| beta 0: lab features add nothing | -0.0045 | within 0.005 | yes |
| beta 0: masking schedules tie | spread 0.0017 | within 0.005 | yes |
| beta 1: lab features add signal | +0.0068 | at least 0.01 | **no** |
| beta 1: keeping labs beats dropping them | +0.0155 | at least 0.01 | yes |

**Reading.** Beta 0 behaves as designed: labs carry nothing and no schedule can matter for prediction.
At beta 1 labs do carry signal (dropping every lab costs 0.016 AUROC, about three standard errors), but
less than the 0.01 margin set for refitting without lab features. One mechanism is visible in the truth
tables: at beta 1, routine labs diagnose 16,311 conditions by the cutoff against 4,165 at beta 0, and
3,291 more conditions are diagnosed by the cutoff overall. Those cases enter the baseline condition
features and leave the at-risk cohort, so the labs' information shows up partly through diagnosis
features rather than lab values, and overall AUROC is lower at beta 1 (0.788 against 0.805). Background
care is also sparse (2.6 visits in seven years), so few pre-cutoff labs fall in the rise window before an
onset. This is a hypothesis to test in the beta sweep (beta 2, denser schedules), not a settled finding.
