# Baseline equity audit (RQ4): notes

Written 2026-09-30, updated 2026-10-12 for the leak-free data (`research/leak_notes.md`). Code:
`research/equity.py` (seed 42, 200 bootstrap resamples). Tables:
`outputs/tables/equity_{monitoring,performance,measured_share}.csv`. Figure:
`outputs/figures/equity_baseline.png`.

Subgroups are sex, age band (<50, 50-64, 65-79, 80+) and insurance. The data has no race or
ethnicity field, so the audit cannot address the subgroups where monitoring gaps are best
documented in real care. The synthetic insurance mix is also implausible (median age 73, but only
18% on Medicare).

## 1. Historical monitoring is flat

Lab visits per patient-year range from 0.365 to 0.369 across every subgroup (largest ratio 1.01),
and the share with no lab before the cutoff is 19% to 20% everywhere. The generator schedules labs
without regard to who the patient is. RQ4's premise, that some subgroups are already less observed,
therefore cannot arise from this data's history. Any inequity found later comes from the policy
itself, which is still worth measuring but is a narrower claim.

## 2. Predictors are fair by sex and insurance, not by age

| Subgroup | GRU AUROC | XGBoost AUROC |
|---|---|---|
| <50 | 0.860 | 0.858 |
| 50-64 | 0.820 | 0.823 |
| 65-79 | 0.787 | 0.797 |
| 80+ | 0.711 | 0.726 |

AUROC differs by less than 0.002 between sexes and at most 0.014 across insurance types. Across age it
falls by 0.13 to 0.15 for both models, so the gradient belongs to the data rather than to either model.
Calibration holds within 3.5 points in every subgroup; the GRU is the less well calibrated of the two
(largest gap 3.5 points by age, 2.9 by insurance, against 1.0 and 1.8 for XGBoost). The oldest patients
are the most likely to progress (63%, against 19% under 50) and the hardest to rank.

## 3. The uncertainty signal decides which age group is measured

Share of each age band measured when the most uncertain patients are measured first, as a multiple of
a fair share (1.0), using the GRU's predictions:

| Rule, budget | <50 | 50-64 | 65-79 | 80+ |
|---|---|---|---|---|
| Dropout SD, 10% | 3.28 | 0.41 | 0.02 | 0.87 |
| Dropout SD, 25% | 1.94 | 0.92 | 0.37 | 1.02 |
| Closeness to 50% risk, 10% | 0.53 | 0.59 | 1.44 | 1.25 |
| Closeness to 50% risk, 25% | 0.59 | 0.99 | 1.07 | 1.22 |

The two signals send tight budgets to different age groups. A dropout-SD threshold rule, the direct
analogue of the sensor-network rule in the proposal, would measure the youngest, lowest-risk patients
three times as often as their share and almost never measure 65-79 year olds. Closeness to 50% risk
favours the over-65s and under-measures the under-50s. Sex stays within 5% of a fair share under both
rules; insurance within 8% under dropout SD and 22% under closeness to 50% risk.

This is the failure mode RQ4 was written to catch, appearing before any policy is learned: the choice of
uncertainty measure alone moves monitoring between age groups. The masking experiments
(`research/scheduling_notes.md`) show the same effect inside a working controller.
