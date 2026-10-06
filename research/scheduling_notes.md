# Scheduling under masking (proposal weeks 7-8): notes

Written 2026-10-05. Code: `research/scheduling.py`. Table: `outputs/tables/scheduling_frontier_masking.csv`.
Figure: `outputs/figures/scheduling_frontier_masking.png`. Seed 42; random controls average five seeds.

## Setup

- Predictor: calibrated XGBoost, as the proposal's week 5 rule requires (the GRU and the Transformer
  both trail it, Tasks 7 and A).
- Evaluation: masking, pending Dr. Aram's decision on the memo. Decision points are each test patient's
  observed pre-cutoff lab visits (10,414 patients, 15,183 visits, at most 8 per patient). A schedule can
  only drop observed visits, never add new ones.
- Threshold controller: at each visit, the risk known from events before that date (including labs the
  controller already kept) decides it; the visit is kept if risk is within tau of 50%.
- Controls: keep all (observed care), keep none, random visits matched on each threshold's budget, and
  fixed 6- and 12-month intervals. The guideline reconstruction is not built yet.
- Spend uses the cost model ($139.76 per measurement) over the four input years.
- Check: the keep-all schedule reproduces the saved baseline AUROC exactly (0.8640); the run stops if not.

## Result: every schedule scores the same

All schedules, from keeping no labs to keeping every lab, fall within 0.0009 AUROC (0.8638 to 0.8646)
and 0.0003 Brier. The random controls vary by about 0.0003 across seeds, so the threshold controller
is indistinguishable from random measurement at every budget, and from no measurement at all.

This is not a masking bug. Removing every lab takes lab tests per patient from 15.05 to 0 and moves
individual predictions (mean change 0.009, largest 0.19), but lab features carry only 3.7% of the
XGBoost's total gain. The model runs on medication counts and indications and condition counts. In this
generator, labs carry almost no information about progression, which matches the Task 9 masking result
for the GRU.

## Equity side

The threshold rule still moves measurement between age groups even though it buys no accuracy. At the
smallest budget (tau 0.02, 3% of visits kept) it keeps patients aged 80 and over at 2.06 times the
overall rate and under-50s at 0.10 times; the pattern holds at every budget (80+ at 1.15 to 2.06 times,
under-50s at 0.10 to 0.70 times). Random and fixed-interval schedules stay within 4% of even. This is
the closeness-to-50% pattern from the baseline equity audit, now in a working controller.

## What it means

- Masking cannot rank schedules on this data: the frontier is flat by construction of the generator.
  This is the strongest evidence yet for the memo's recommendation.
- The proposal's "delay in detecting a new condition" metric is not computable under masking, because
  diagnoses are not produced by labs here. It needs a simulator or real data.
- The equity finding stands on its own: an uncertainty-threshold rule can shift who is monitored
  without improving prediction, which is the failure RQ4 is meant to catch.
