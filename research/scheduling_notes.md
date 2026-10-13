# Scheduling under masking (proposal weeks 7-8): notes

Updated 2026-10-12 for the leak-free data (`research/leak_notes.md`); the first run on the original data
was flat for a reason that turned out to be the medication leak. Code: `research/scheduling.py`. Table:
`outputs/tables/scheduling_frontier_masking.csv`. Figure: `outputs/figures/scheduling_frontier_masking.png`.
Seed 42; random controls average five seeds.

## Setup

- Predictor: calibrated XGBoost on leak-free features, as the proposal's week 5 rule requires.
- Evaluation: masking. Decision points are each test patient's observed pre-cutoff lab visits (10,414
  patients, 15,183 visits, at most 8 per patient). A schedule can only drop observed visits.
- Threshold controller: at each visit, the risk known from events before that date (including labs the
  controller already kept) decides it; the visit is kept if risk is within tau of 50%.
- Controls: keep all (observed care), keep none, budget-matched random visits, fixed 6- and 12-month
  intervals. The guideline reconstruction is not built yet.
- Spend uses the cost model ($139.76 per measurement) over the four input years.
- Check: the keep-all schedule reproduces the saved calibrated baseline AUROC exactly; the run stops if not.

## Results

Labs now matter: keeping none scores AUROC 0.817 and keeping all 0.828.

| Schedule | Visits kept | Spend per patient-year | AUROC | Random at same budget (SD) |
|---|---|---|---|---|
| Threshold, tau 0.1 | 15% | $7.9 | 0.8203 | 0.8197 (0.0011) |
| Threshold, tau 0.15 | 24% | $12.3 | 0.8223 | 0.8210 (0.0010) |
| Threshold, tau 0.2 | 38% | $19.5 | 0.8250 | 0.8229 (0.0009) |
| Threshold, tau 0.3 | 74% | $37.7 | 0.8271 | 0.8261 (0.0004) |
| Fixed 12-month | 78% | $39.8 | 0.8281 | |
| Fixed 6-month | 88% | $44.8 | 0.8281 | |
| Keep all | 100% | $50.9 | 0.8276 | |

- **Against random**, the threshold controller wins at mid budgets: about 2 standard deviations of the
  random controls at tau 0.2 and 0.3, with matching Brier gains. The margin is small (0.001 to 0.002
  AUROC), and five seeds give only a rough spread, so this is suggestive rather than settled.
- **Against fixed intervals**, the proposal's RQ2 comparison, it loses: a 12-month schedule at $39.8
  scores 0.8281, above the threshold controller at a similar $37.7. Dropping visits within a year of the
  previous one costs nothing, because closely spaced visits repeat information.
- Masking can only test schedules sparser than observed care, and the fixed-interval budgets cannot be
  matched exactly, so the comparison is between nearby points rather than at equal spend.

## Equity side

The controller moves measurement between age groups, and the direction depends on the budget. As a
multiple of a fair share:

| tau | <50 | 50-64 | 65-79 | 80+ |
|---|---|---|---|---|
| 0.1 | 1.06 | 2.45 | 0.17 | 0.51 |
| 0.15 | 0.81 | 1.83 | 1.03 | 0.40 |
| 0.2 | 0.74 | 1.33 | 1.25 | 0.67 |
| 0.3 | 0.74 | 1.05 | 1.12 | 1.02 |

Fixed-interval schedules stay within 0.4% of even in every age group, and random schedules within 1% at
budgets of 15% and above. At the tiniest budget (0.2% of visits, about 33 visits) random selection drifts
by up to 23% by chance alone, so the threshold rule's shifts at the tight budgets above are compared
against that noise. The threshold rule's age effect is
large at tight budgets and is not simply "favours the old" or "favours the young": it follows where
predicted risk sits near 50%, which shifts as labs are dropped.

## What it means

- RQ2 under masking: an uncertainty-threshold rule beats random measurement slightly but not a plain
  annual schedule. That is a real, reportable result on this data, with the caveat that masking cannot
  add measurements.
- RQ3's detection-delay metric is still not computable under masking; it needs the simulator.
- RQ4: the threshold rule redistributes monitoring by age much more than its accuracy gain would justify.
