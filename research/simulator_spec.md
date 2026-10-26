# Semi-synthetic multimorbidity simulator: specification

Written 2026-10-12, ISEM 735 week 6, and updated the same day for the leak-free data. Decisions
**[D1]** to **[D4]** were approved by Chinmay on 2026-10-12.

## Why a simulator, and why not one learned from this data

Updated 2026-10-12. On the leak-free data (`research/leak_notes.md`) masking is no longer flat: keeping no
labs scores AUROC 0.817 and keeping all 0.828, and an uncertainty-threshold rule beats random measurement
slightly but not a fixed 12-month schedule (`research/scheduling_notes.md`). Masking still has three
limits a simulator removes. It can only drop observed visits, so it cannot test schedules denser than
care in the data. It cannot compute the delay in detecting a new condition (RQ3), because nothing in the
record says when a condition truly began. And with no ground truth it cannot check fitted Q-evaluation.

A generator learned from this data (for example a Delphi-style transformer) would also learn the Kaggle
generator's artifacts: labs that track all-time diagnosis flags regardless of timing, and medications
that start before their diagnoses. The simulator instead gives each lab-linked condition a hidden
severity that the labs observe with a tunable signal strength, beta, and starts medications at
diagnosis. Everything else is calibrated to the Kaggle data. At beta = 0 the labs carry no information,
which is the simulator's first sanity check: every schedule should then score the same. Sweeping beta
upward asks the question the data alone cannot: how informative do labs have to be before adaptive
scheduling beats fixed intervals, and for whom?

**Lab magnitudes.** In the Kaggle data, labs differ strongly between patients ever and never diagnosed
(creatinine 5.74 against 0.90 for kidney disease; LDL 142 against 64 for hyperlipidemia; glucose 170
against 131 for diabetes; no difference for BNP). At beta = 1 and full severity the simulated lab means
reach those diagnosed means, so magnitudes come from the data and timing comes from the literature.

Testing evaluation methods inside a simulator with known ground truth has precedent: Oberst and Sontag
(2019) used a synthetic sepsis environment to examine off-policy evaluation of treatment policies.

## Model

**Population.** Each simulated patient is resampled from the Kaggle patients table (age, sex, insurance,
BMI, smoking, conditions present at entry), so covariate mixes match the data.

**Time.** Monthly steps over 2018-2024 (84 months). A schedule decides each month whether to order a
measurement (lab panel plus office visit, $139.76 from the cost model). **[D1]** monthly steps, chosen
for a clean decision process; continuous time is possible but slower to build.

**Lab-linked conditions.** **[D2]** Type 2 diabetes (fasting glucose, HbA1c), chronic kidney disease
(creatinine, eGFR), heart failure (BNP) and hyperlipidemia (lipid panel). Each has a hidden severity
s(t) that drifts upward at a rate depending on age, sex and other conditions, with noise. True onset is
the month s first reaches 1.

- *Diabetes trajectory.* In Whitehall II, fasting glucose rose linearly and then steeply over the 3 years
  before diagnosis, from 5.79 to 7.40 mmol/L, with changes detectable 3 to 6 years before (Tabák et al.,
  2009). The glucose and HbA1c mapping from s is set so that beta = 1 reproduces this trajectory.
- *Kidney trajectory.* Mean creatinine-based eGFR loss in adults aged about 72 was 0.4 mL/min/1.73 m^2
  per year, with 16% losing more than 3 per year (Shlipak et al., 2009); in healthy adults the normal
  decline is about 0.4 to 1.1 per year (Guppy et al., 2024). The drift spread is set to give a similar
  share of fast decliners.
- *Heart failure.* In older adults with low baseline levels, a rise in NT-proBNP over 2 to 3 years
  marked much higher 10-year heart failure incidence (Glick et al., 2013). The BNP mapping rises with s
  so that increases precede diagnosis by a similar window.

**Other conditions.** The 11 conditions without a linked lab (asthma, depression and so on) start at
hazards fitted to the Kaggle condition-onset rates by age band and sex. Measurement does not change them.

**Labs.** When a measurement is ordered, the panel returns each test as
value = normal mean + beta x (signal from s) + noise, with normal means and noise taken from the Kaggle
lab distributions and binned low, normal or high against the data's reference ranges. Beta = 0 gives labs
with no information about s, which is the Kaggle regime.

**Diagnosis.** A lab-linked condition is diagnosed at whichever comes first: a measurement whose value
crosses the diagnostic threshold, or clinical presentation. Without measurement, diabetes can go
undiagnosed for years: retinopathy prevalence suggests onset 9 to 12 years before clinical diagnosis,
with detectable damage 4 to 7 years before (Harris et al., 1992). Clinical presentation is therefore
modelled as s reaching a higher level, set so the unmeasured lag from onset is several years. Detection
delay (diagnosis month minus true onset month) is the RQ3 metric that masking could not compute.

**Medications** start at diagnosis by indication, at the Kaggle rates. **[D4]** Hospital admissions are
left out of v1; they matter for the hospitalization target, which has almost no signal in this data.

**Historical observation.** Background care orders labs at the Kaggle rate (0.37 lab visits per
patient-year, random timing). For RQ4, an option lowers that rate for one subgroup, so the audit can test
whether a policy amplifies existing under-observation; the Kaggle data has none to test.

**Output.** The same five-table schema as the Kaggle data, so `src/sequences.py`, `src/features.py`,
the predictors, the cost model and the equity audit all run unchanged on simulated cohorts.

**[D3]** Design: v1 is mechanistic (transparent, fast, every assumption written down). A learned
component, such as a sequence model that generates the non-lab conditions, is a stretch goal; the
"more AI" work lands in the retrained predictor and the week 9 learned policy instead.

## W6-1 implementation decisions (confirmed by Chinmay, 2026-10-26)

The core (`research/simulator.py`) refines the model above with "resample and overlay": each simulated
patient copies a resampled Kaggle patient's covariates, admissions, background lab-visit dates and
non-lab-linked diagnoses, and only the four lab-linked conditions, their labs and their medications are
regenerated. Admissions are copied unchanged (refines D4) so the pipeline's schema holds. If the source
patient was diagnosed, that date is the clinical diagnosis and true onset is a lag before it; others can
have an onset whose clinical diagnosis would fall after 2024, at the data's incidence. Severity rises
quadratically over the window W before onset.

| Item | Decision |
|---|---|
| Window W (months) | diabetes 36, kidney 48, heart failure 30, hyperlipidemia 24 (judgement) |
| Clinical lag (months, uniform) | diabetes 24-72, kidney 24-60, heart failure 6-24; hyperlipidemia: onset between the previous routine lab visit and the clinical diagnosis (judgement) |
| Panel | every lab visit returns all 17 tests; the data's test choice gives the diagnosis away (HDL and triglycerides only in hyperlipidemics, HbA1c 96% in diabetics, eGFR 90% in kidney patients) |
| Tests without contrast | HbA1c healthy 5.4% (SD 0.5) and BNP healthy from the data; both reach the cutoff plus one healthy SD at full severity (HbA1c 7.0, BNP 116.6). HDL and triglycerides carry no signal |
| Triggers | clinical cutoffs: glucose >= 126, HbA1c >= 6.5, eGFR < 60, LDL >= 160, BNP > 100; diagnose only if s >= 1 |
| Noise | the data's never-diagnosed spread per test (glucose SD 52) |

Consequence of the cutoffs plus the data's wide "healthy" glucose (45% of never-diabetic values are at or
above 126): even at beta = 0 a measurement after onset triggers a correct diabetes diagnosis about 45% of
the time, so measuring more often shortens diabetes detection delay at beta = 0. That beta = 0 delay is
the baseline RQ3 effects are measured against. In the 5,000-patient sample (seed 42), 301 of 729
diagnosed diabetics were lab-detected at beta = 0 and 445 of 769 at beta = 1.

## Validation, before any experiment

At beta = 0 the simulator must match the Kaggle data on:

| Target | Kaggle value |
|---|---|
| Lab visits per patient over 7 years | 2.58 (median 2) |
| Median gap between lab visits | 431 days |
| Multimorbidity progression in the at-risk cohort | 43.5% (test set) |
| Correlation of lab visits with Charlson index | 0.00 |
| Masking frontier at beta = 0 | flat (labs carry no information) |

Plus condition prevalence and onset by age band, and lab abnormal rates by test, each within a tolerance
set in the tests.

## Experiments the simulator enables (weeks 7 to 10)

- RQ2 and RQ3: threshold controller (dropout spread and closeness to 50% risk), fixed 3-, 6- and 12-month
  schedules, budget-matched random, and a guideline-based schedule, across beta in {0, 0.25, 0.5, 1, 2}.
  Report spend, detection delay, and predictor AUROC and Brier.
- Week 9: a learned policy trained in the simulator, then its value estimated from logged simulated data
  with fitted Q-evaluation and compared with the simulator's exact value. This keeps the proposal's
  committed estimator and tests it against ground truth.
- RQ4: age-band and insurance measurement shares for every policy, with and without injected subgroup
  under-observation.

## Limits to state in the report

- Results show mechanism, not clinical magnitude. Beta = 1 is anchored to published trajectories, but the
  simulator's structure is an assumption.
- Severity drifts independently per condition apart from shared risk factors; real comorbidities
  interact more.
- The data has no race or ethnicity field, so neither does the simulator.

## References

Glick, D., deFilippi, C. R., Christenson, R., Gottdiener, J. S., & Seliger, S. L. (2013). Long-term
trajectory of two unique cardiac biomarkers and subsequent left ventricular structural pathology and
risk of incident heart failure in community-dwelling older adults at low baseline risk. *JACC: Heart
Failure, 1*(4), 353-360. https://doi.org/10.1016/j.jchf.2013.04.007

Guppy, M., Thomas, E. T., Glasziou, P., Clark, J., Jones, M., O'Hara, D. V., & Doust, J. (2024). Rate of
decline in kidney function with age: A systematic review. *BMJ Open, 14*(11), e089783.
https://doi.org/10.1136/bmjopen-2024-089783

Harris, M. I., Klein, R., Welborn, T. A., & Knuiman, M. W. (1992). Onset of NIDDM occurs at least 4-7 yr
before clinical diagnosis. *Diabetes Care, 15*(7), 815-819. https://doi.org/10.2337/diacare.15.7.815

Oberst, M., & Sontag, D. (2019). Counterfactual off-policy evaluation with Gumbel-max structural causal
models. In *Proceedings of the 36th International Conference on Machine Learning* (Vol. 97,
pp. 4881-4890). PMLR.

Shlipak, M. G., Katz, R., Kestenbaum, B., Fried, L. F., Newman, A. B., Siscovick, D. S., Stevens, L., &
Sarnak, M. J. (2009). Rate of kidney function decline in older adults: A comparison using creatinine and
cystatin C. *American Journal of Nephrology, 30*(3), 171-178. https://doi.org/10.1159/000212381

Tabák, A. G., Jokela, M., Akbaraly, T. N., Brunner, E. J., Kivimäki, M., & Witte, D. R. (2009).
Trajectories of glycaemia, insulin sensitivity, and insulin secretion before diagnosis of type 2
diabetes: An analysis from the Whitehall II study. *The Lancet, 373*(9682), 2215-2221.
https://doi.org/10.1016/S0140-6736(09)60619-X
