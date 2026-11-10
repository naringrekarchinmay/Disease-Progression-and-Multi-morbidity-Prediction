# Pre-registration: simulator sensitivity sweep

Written 2026-11-09, before any run below. The commit that adds this file is the timestamp; the sweep code
must be committed after it. Prompted by Part 3 feedback, major issue 4 (a simulator whose settings decide the
answer). Dr. Aram approved the simulator as the main evaluation on these terms at the 2026-11-03 meeting.

## What was already run, and what that means for this sweep

Disclosed so a reader can judge what was chosen after seeing results:

- The beta grid {0, 0.25, 0.5, 1, 2} and the beta = 0 sanity check were fixed in `simulator_spec.md` on
  2026-10-12, before any policy ran.
- The onset window W, the clinical lag and the hyperlipidemia onset rule were fixed on 2026-10-26 (W6-1),
  before any policy ran, but were never varied.
- The headline lambda was **moved from $50 to $200 after** task 19 showed never measuring wins at $50. This
  sweep therefore reports every lambda in the grid and leads with break-even lambda, not one value.
- All results so far use one seed (42 evaluation, 43 training, 44 threshold calibration).

## Fixed before running

**Simulator code**: `research/simulator.py` as of this commit. Nothing in it changes during the sweep.

**Parameters (one at a time around the base case, each crossed with every beta):**

| Factor | Levels | Base |
|---|---|---|
| beta | 0, 0.25, 0.5, 1, 2 | all |
| Onset window W, all four conditions scaled together | 0.5x, 1x, 1.5x of (diabetes 36, kidney 48, heart failure 30, hyperlipidemia 24 months), rounded to whole months | 1x |
| Clinical lag, both ends of each range scaled together | 0.5x, 1x, 1.5x of (diabetes 24-72, kidney 24-60, heart failure 6-24 months); hyperlipidemia rule unchanged | 1x |
| lambda (price of an undiagnosed condition-month) | $25, $50, $100, $200, $400 | reported at every level |

That is 5 base cells plus 2 x 5 for W plus 2 x 5 for lag: 25 simulator settings.

**Seeds**: three replicates per setting. Replicate r uses evaluation cohort seed 42 + 1000r, training seed
43 + 1000r and threshold seed 44 + 1000r, for r = 0, 1, 2 (r = 0 reproduces the existing runs).

**Cohort sizes**: unchanged from tasks 16-21 (100,000 evaluation patients; 30,000 training patients).
Pre-stated fallback if the full grid does not finish by 2026-11-12: the W and lag settings drop to one
replicate (r = 0) and evaluation cohorts to 50,000; the base cells keep three replicates. Either way the
choice is made on run time, before looking at results.

**Policies scored in every setting**: never measure, observed care, guideline, fixed 12-month, fixed 3-month,
and the fitted Q-iteration policy retrained in that setting at lambda $100, $200 and $400 (task 20 code,
unchanged settings). The oracle is reported for scale only. Fixed policies need no retraining, because their
value is linear in lambda.

## Outcomes

- **Primary**: true discounted value per patient of the learned policy minus the best fixed policy in that
  setting (best of never, observed, guideline, 12-month, 3-month), at each lambda the learned policy was
  trained for.
- **Secondary**: restricted mean detection delay (months) per policy; break-even lambda of each fixed policy
  against never measuring; measurements per patient-year.
- Interval: mean over the three replicates, with a 95% CI from a patient-level bootstrap pooled across
  replicates (2,000 resamples, `research/stats_ci.py`).

## Decision rules, fixed now

1. **Adaptive wins in a setting** when the lower 95% bound of the primary outcome is above $0.
2. **A win is robust** when it holds at every W and lag level for that beta and lambda. The report presents
   the robust region as a map over beta and lambda, not as one number.
3. **Sanity check**: at beta = 0 and lambda up to $200 the learned policy should not win (labs carry no
   information). A win there is reported as a failure of the check, not as a finding. The known $400 win at
   beta 0 (task 20) is reported as such.
4. **Conclusion wording**: the headline claim is conditional, of the form "a learned policy beats every
   fixed schedule when beta is at least B and an undiagnosed month costs at least L", with B and L read off
   rule 2. If no robust region exists, that is the result.

## Reporting choices, fixed now

- **Hyperlipidemia**: its onset rule makes early detection depend on extra measurement by construction, so
  the primary outcome is also reported with hyperlipidemia's undiagnosed months removed from the reward. If
  the two disagree on rule 1 in any cell, the report says so.
- **Beta scale**: beta = 1 lab magnitudes are the Kaggle diagnosed means, which inherit the lab artifact
  (task 22). Beta is therefore reported as a relative scale of lab informativeness, not as a clinical
  quantity; no re-anchoring is done in this sweep.

## Deviations

Any change after this commit is listed here with its date and reason, before results are reported.

| Date | Change | Reason |
|---|---|---|
| 2026-11-09 | Added a supplementary interval: a t interval over the three replicates (`wins_by_replicate_interval`), reported next to the pre-registered rule, which is unchanged. | After the run, the patient-level bootstrap CIs (about +/- $5) proved far narrower than the spread between replicates (SD $3 to $100), because the bootstrap leaves out the variance of retraining the learned policy. Added after seeing results, so it is labelled supplementary. |
