# Baseline equity audit (RQ4): notes

Written 2026-09-30. Code: `research/equity.py` (seed 42, 200 bootstrap resamples). Tables:
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

| Subgroup | GRU AUROC | XGBoost AUROC | Progression rate |
|---|---|---|---|
| <50 | 0.916 | 0.929 | 19% |
| 50-64 | 0.868 | 0.877 | 35% |
| 65-79 | 0.827 | 0.838 | 49% |
| 80+ | 0.739 | 0.750 | 63% |

AUROC differs by less than 0.01 between sexes and about 0.02 across insurance types, with
overlapping intervals. Across age it falls by 0.18 for both models, so the gradient belongs to the
data rather than to either model. Calibration holds in every subgroup (largest gap 1.9 points,
the GRU under-predicting uninsured patients). The oldest patients are the most likely to progress
and the hardest to rank.

## 3. The uncertainty signal decides which age group is measured

Share of each age band measured when the most uncertain patients are measured first, as a
multiple of a fair share (1.0):

| Rule, budget | <50 | 50-64 | 65-79 | 80+ |
|---|---|---|---|---|
| Dropout SD, 10% | 2.63 | 0.97 | 0.31 | 0.56 |
| Dropout SD, 25% | 1.44 | 1.12 | 0.76 | 0.82 |
| Closeness to 50% risk, 10% | 0.51 | 0.97 | 1.03 | 1.32 |
| Closeness to 50% risk, 25% | 0.65 | 0.94 | 1.10 | 1.20 |

The two signals send tight budgets to opposite ends of the age range. A dropout-SD threshold rule,
the direct analogue of the sensor-network rule in the proposal, would measure the youngest,
lowest-risk patients most and the oldest, highest-risk and least accurately predicted patients
least. Sex and insurance stay within about 10% of a fair share under both rules.

This is the failure mode RQ4 was written to catch, appearing before any policy is learned: the
choice of uncertainty measure alone moves monitoring between age groups. The threshold controller
in weeks 7 and 8 should report age-band selection rates beside its budget frontier, and the
comparison between the two uncertainty signals belongs in the results.
