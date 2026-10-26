# Beta sweep, part 1: non-adaptive schedules (W7-1, task 16)

Written 2026-10-26. Code: `research/simulator_sweep.py`. Tests: `tests/test_simulator_sweep.py`.
Table: `outputs/tables/simulator_sweep_schedules.csv` (100,000 simulated patients, seed 42; 178 s).

## Setup (decisions confirmed by Chinmay, 2026-10-26)

- A schedule is the patient's full set of lab visits; it **replaces** background care. Observed care (the
  resampled patients' own Kaggle visit dates) is one schedule. Each visit costs $214.49 (full panel plus
  office visit, `research/cost_model.md`).
- Latent truth is the same as the W6 cohorts' for seed 42 (tested). Lab noise uses common random numbers
  keyed by patient, month and test, so schedules measuring the same patient-month see the same value.
- Fixed schedules start in a random month per patient; 3-, 6- and 12-month schedules share that draw and
  are nested. Random visits are matched to the 12-month total (mean of 3 seeds).
- **Guideline schedule**, verified against the primary sources:
  - yearly once diabetes or kidney disease is known: "Both albuminuria and eGFR should be monitored
    annually" (ADA, 2025b, Section 11; with established CKD, 1 to 4 times a year, so yearly is the floor);
  - otherwise every 3 years from age 35: ADA (2025a, Section 2) recommends screening from age 35 and repeat
    screening "at a minimum of 3-year intervals"; USPSTF (2021) says every 3 years "may be a reasonable
    approach" for adults 35 to 70 with overweight or obesity;
  - otherwise every 5 years from age 20: ACC/AHA (Goff et al., 2014) finds it reasonable to assess
    traditional risk factors, cholesterol included, every 4 to 6 years in adults 20 to 79;
  - no routine panel under 20. No guideline sets a heart failure screening interval, so heart failure is
    detected only incidentally by the guideline's visits. The interval is set at each visit from what is
    known then; schedule and detection are iterated until the schedule stops changing.
- Prediction metrics (AUROC, Brier) are deliberately left to the next task, scored against labels from true
  onsets, because a denser schedule changes the diagnosis-based labels and the at-risk cohort.

**Delay** is months from true onset to diagnosis, over onsets inside 2018-2024 (12,391 kidney, 9,481
diabetes, 1,931 heart failure, 40,436 hyperlipidemia); an onset still undiagnosed at the end of 2024 counts
as diagnosed at month 84 (restricted mean). **False-alarm visits** are visits where a trigger fired for a
condition that had not begun; the confirmatory work they would cause is not priced.

## Results: all four conditions

Restricted mean delay in months (spend per patient-year in brackets).

| Schedule | beta 0 | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|
| Observed care ($79) | 15.1 | 14.7 | 13.6 | 11.9 | 11.6 |
| Guideline ($112-124) | 14.9 | 14.4 | 12.8 | 10.1 | 8.8 |
| 12-month ($214) | 13.5 | 12.7 | 9.9 | 6.5 | 4.5 |
| Random, matched to 12-month ($214) | 13.9 | 13.2 | 10.7 | 8.0 | 6.5 |
| 6-month ($429) | 12.7 | 11.5 | 8.0 | 4.6 | 2.4 |
| 3-month ($858) | 12.1 | 10.3 | 6.3 | 3.0 | 1.1 |

Share detected by labs at beta 1: observed care 23%, guideline 39%, 12-month 48%, 3-month 72%.
False-alarm visits per patient-year: 0.17 observed care, 0.22-0.27 guideline, 0.47-0.51 12-month,
1.9-2.0 3-month, almost all from fasting glucose.

## By condition (restricted mean delay, months)

| Condition | Schedule | beta 0 | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|
| Kidney disease | observed care | 29.1 | 28.2 | 23.3 | 17.4 | 17.4 |
| | guideline | 29.1 | 28.2 | 22.2 | 13.8 | 13.7 |
| | 12-month | 29.1 | 26.8 | 15.1 | 5.3 | 5.2 |
| | 3-month | 28.9 | 21.2 | 4.9 | 1.0 | 1.0 |
| Type 2 diabetes | observed care | 22.6 | 21.4 | 20.0 | 18.0 | 17.7 |
| | guideline | 21.2 | 19.6 | 17.2 | 14.0 | 13.5 |
| | 12-month | 12.1 | 10.2 | 8.0 | 5.5 | 5.2 |
| | 3-month | 3.3 | 2.6 | 1.9 | 1.1 | 1.0 |
| Heart failure | observed care | 13.2 | 13.2 | 12.8 | 10.9 | 10.5 |
| | 12-month | 13.2 | 13.1 | 11.8 | 6.4 | 5.2 |
| Hyperlipidemia | observed care | 9.2 | 9.1 | 9.1 | 8.9 | 8.4 |
| | 12-month | 9.1 | 9.0 | 8.7 | 7.1 | 4.1 |

## Reading

1. **At beta 0, kidney disease, heart failure and hyperlipidemia are flat across schedules** (within 0.3
   months), as they should be when labs carry no information. Measuring more often does not help.
2. **Diabetes is the exception at beta 0**: delay falls from 22.6 months (observed care) to 3.3 (3-month).
   This is the W6 glucose effect at full strength. Kaggle's "healthy" glucose crosses 126 mg/dL 45% of the
   time, so any test after onset has a coin-flip chance of a correct diagnosis whatever beta is. RQ3 effects
   for diabetes must be read against this beta 0 baseline, and it is the main source of false alarms.
3. **Regular spacing beats random at equal spend**: the 12-month schedule's delay is 1.5 months shorter than
   budget-matched random at beta 1 (6.5 against 8.0) and 2.0 at beta 2. Random visits cluster and leave
   gaps; this is the bar an adaptive policy has to clear on timing alone.
4. **The guideline spends 1.4 to 1.6 times observed care** and buys most of its gain once disease is known
   (kidney 17.4 to 13.8 months at beta 1). A flat 12-month schedule at 1.7 to 1.9 times the guideline's cost
   cuts delay by a further 3.6 months at beta 1.
5. **Saturation above beta 1 for kidney disease and diabetes**: at s = 1 and beta 1, mean eGFR is already
   34.6 (cutoff 60), so a measurement after onset almost always triggers; beta 2 adds nothing. Heart failure
   and hyperlipidemia keep improving because their full-severity means sit near the cutoff (BNP 117 against
   100; LDL 142 against 160).
6. **Equity (RQ4)**: fixed and random schedules measure every age band equally. The guideline measures
   under-50s 10 to 12% less than average and over-80s 4 to 5% more, through its age-35 threshold. Delay
   differences by age band are under 0.6 months for every schedule at beta 1.

## Limits

- Hyperlipidemia's delay is short under every schedule because its onset is placed after the previous
  routine visit by construction (W6-1); its numbers say little about scheduling.
- False alarms cost a visit in the model but nothing more; confirmatory testing is not priced.
- The guideline is a simplified composite. It ignores BMI, hypertension and other risk factors that
  change real screening intervals.

## References

American Diabetes Association Professional Practice Committee. (2025a). 2. Diagnosis and classification of
diabetes: Standards of Care in Diabetes-2025. *Diabetes Care, 48*(Supplement 1), S27-S49.
https://doi.org/10.2337/dc25-S002

American Diabetes Association Professional Practice Committee. (2025b). 11. Chronic kidney disease and risk
management: Standards of Care in Diabetes-2025. *Diabetes Care, 48*(Supplement 1), S239-S251.
https://doi.org/10.2337/dc25-S011

Goff, D. C., Jr., Lloyd-Jones, D. M., Bennett, G., Coady, S., D'Agostino, R. B., Gibbons, R., Greenland, P.,
Lackland, D. T., Levy, D., O'Donnell, C. J., Robinson, J. G., Schwartz, J. S., Shero, S. T., Smith, S. C., Jr.,
Sorlie, P., Stone, N. J., & Wilson, P. W. F. (2014). 2013 ACC/AHA guideline on the assessment of cardiovascular
risk: A report of the American College of Cardiology/American Heart Association Task Force on Practice
Guidelines. *Journal of the American College of Cardiology, 63*(25 Pt B), 2935-2959.
https://doi.org/10.1016/j.jacc.2013.11.005

US Preventive Services Task Force. (2021). Screening for prediabetes and type 2 diabetes: US Preventive
Services Task Force recommendation statement. *JAMA, 326*(8), 736-743. https://doi.org/10.1001/jama.2021.12531

---

# Beta sweep, part 2: prediction under each schedule (W7-2, task 17)

Added 2026-10-26. Code: `research/simulator_prediction.py`. Tests: `tests/test_simulator_prediction.py`.
Table: `outputs/tables/simulator_sweep_prediction.csv` (100,000 patients, seed 42; 28 minutes).

## Setup

- **Labels from truth, the same for every schedule.** A patient is at risk if fewer than two conditions have
  truly begun by 2021-12-31 (true onset for the four lab-linked conditions, the copied diagnosis date for
  the other 11), and progresses if two or more have begun by 2024-12-31. True at-risk cohort 38,553;
  progression 51.6% (against 44.4% on diagnosis-based labels, because undiagnosed onsets now count).
- **Features are what the schedule saw by the cutoff**: its lab visits up to 2021-12-31, the diagnoses those
  visits and clinical presentation produced, and medications from diagnosis. They are built with the
  unchanged `src/features.py`, in 20,000-patient chunks (tested equal to one pass).
- **One calibrated XGBoost per schedule** (a model deployed under a schedule is trained on data collected
  under it), all on one patient split (test 9,639), so AUROC differences are paired. Standard errors of
  differences are a 200-draw bootstrap of the test patients; they leave out model-refit variance, so they
  understate the true uncertainty.
- Added a **no labs** schedule (no measurements at all) as the floor. Random visits use one seed here.
  Spend counts pre-cutoff visits only (2018-2021), so the guideline's figure differs from part 1's.

## Results

AUROC (Brier in brackets; spend per patient-year over 2018-2021).

| Schedule | spend | beta 0 | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|
| No labs | $0 | 0.748 (0.204) | 0.748 | 0.748 | 0.748 | 0.748 |
| Observed care | $79 | 0.742 (0.206) | 0.752 | 0.754 | 0.757 (0.200) | 0.760 |
| Guideline | $116-131 | 0.742 | 0.765 | 0.772 | 0.777 (0.192) | 0.784 |
| 12-month | $214 | 0.738 | 0.764 | 0.772 | 0.776 (0.192) | 0.785 |
| Random, matched to 12-month | $215 | 0.741 | 0.760 | 0.765 | 0.769 (0.195) | 0.774 |
| 6-month | $429 | 0.738 | 0.768 | 0.774 | 0.782 (0.190) | 0.793 |
| 3-month | $858 | 0.739 | 0.771 | 0.779 | 0.786 (0.188) | 0.800 |

Gain over observed care at beta 1 (bootstrap SE): 3-month +0.029 (0.003), 6-month +0.025 (0.003),
12-month +0.019 (0.002), guideline +0.020 (0.003), random +0.011 (0.002). Calibration slopes stay between
0.93 and 1.00 throughout.

## Reading

1. **At beta 0 every measured schedule scores slightly below no labs** (0.006 to 0.010 AUROC, against a
   bootstrap SE near 0.0015). The schedules agree with each other within 0.004. Uninformative lab columns
   cost the untuned XGBoost a little; the W6-3 smoke run showed the same (0.0045). It is a reason to keep
   the hyperparameter search planned for week 8, not evidence that labs mislead.
2. **From beta 0.25 on, measuring more predicts better, with diminishing returns.** At beta 1, going from
   observed care to 12-month buys 0.019 AUROC for $135 more per year; from 12-month to 3-month buys 0.010
   more for $644.
3. **The guideline matches the 12-month schedule at about 60% of its cost** (beta 1: 0.777 against 0.776,
   $131 against $214). It spends its visits on people with known diabetes or kidney disease and on
   over-35s.
4. **Regular spacing beats random spacing at the same spend** (beta 1: 0.776 against 0.769; beta 2: 0.785
   against 0.774), consistent with part 1's detection delays.
5. Even the 3-month schedule at beta 2 reaches only 0.800. Most of the progression signal sits in the other
   11 conditions and the covariates, which no schedule changes.

## Limits

- One seed, one split and default XGBoost settings. Differences of 0.005 or less should not be read as real.
- Random visits use one seed for prediction (three in part 1).
- Medication templates for newly lab-detected patients are drawn per chunk, so medication features can
  differ slightly between schedules for the same patient.
