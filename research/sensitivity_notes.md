# Pre-registered sensitivity sweep: results

Written 2026-11-09 (task 26). Plan: `research/preregistration.md` (commit 3d4c69c), code
`research/simulator_sensitivity.py` (commit 377385b), both committed before any run. Tables:
`outputs/tables/simulator_sensitivity.csv` (primary outcome and break-even lambda) and
`simulator_sensitivity_secondary.csv` (delay, measurements). Per-patient components under
`data/processed/simulated/sensitivity/` (gitignored). Answers Part 3 feedback, major issue 4.

## What ran

All 25 settings (5 betas; onset window W at 0.5x, 1x, 1.5x; clinical lag at 0.5x, 1x, 1.5x; one factor at
a time) with all three replicates, 100,000 evaluation and 30,000 training patients each: 75 runs in about
40 minutes over three processes, so the run-time fallback was not needed. **Replicate 0 of the base case
reproduces task 20 exactly** (every fixed and learned policy value at every beta and lambda).

## Primary outcome, base case (learned policy minus best fixed policy, $ per patient, 3 replicates pooled)

| beta | lambda $100 | $200 | $400 |
|---|---|---|---|
| 0 | -92 (never) | -62 (never) | +267 (never) |
| 0.25 | -138 (never) | +8 (never) | +428 (guideline) |
| 0.5 | +124 (never) | +677 (guideline) | +1,122 (12-month) |
| 1 | +345 (never) | +608 (guideline) | +1,035 (12-month) |
| 2 | +485 (never) | +811 (guideline) | +1,369 (12-month) |

Best fixed policy in brackets. Patient-bootstrap 95% CIs are about +/- $2 to $13 around each value.

## Pre-registered rule 2: where the learned policy wins at every W and lag level

| beta | $100 | $200 | $400 |
|---|---|---|---|
| 0 | no | no | no |
| 0.25 | no | no | no |
| 0.5 | no | **yes** | **yes** |
| 1 | yes (no without hyperlipidemia) | **yes** | **yes** |
| 2 | yes (no without hyperlipidemia) | **yes** | **yes** |

**Pre-registered conclusion (rule 4).** A learned policy beats every fixed schedule, robustly across the onset
window and clinical lag ranges, **when beta is at least 0.5 and an undiagnosed condition-month costs at least
$200**. At $100 it also wins from beta 1, but only when hyperlipidemia counts in the reward; without it, $100
is never enough. Bold cells hold under both rewards.

## Rule 3, the sanity check: one failure

At beta 0 and lambda up to $200 the learned policy should not win. It does not in 9 of 10 cells, but with
the clinical lag at 1.5x it wins at $200 by $139 (bootstrap CI $131 to $148). Pre-registered handling: this
is a **failure of the check**, not a finding. Likely cause, consistent with task 16: at beta 0 a routine
glucose test still detects diabetes about 45% of the time (the Kaggle healthy glucose spread), so measuring
has value even when labs carry no severity signal, and a longer lag means more undiagnosed months to save.
The $400 wins at beta 0 (all W levels and base) were disclosed in advance and have the same cause.

## Supplementary (added after the run; see the Deviations table)

The patient bootstrap understates uncertainty: its CIs are about +/- $5, while the three replicates differ
with an SD of $3 to $100, because retraining the learned policy on a new training cohort changes it. A t
interval over replicates (2 degrees of freedom) is stricter:

| beta | $100 | $200 | $400 |
|---|---|---|---|
| 0 to 0.25 | no | no | no |
| 0.5 | no | no | yes |
| 1 | no | yes | yes |
| 2 | yes (all conditions only) | yes | yes |

Under it the sanity-check failure disappears (lower bound -$4.6), and the safe claim narrows to **beta at
least 1 and lambda at least $200, or beta 0.5 at $400**. Recommended wording for the report: lead with the
pre-registered region, and state that the conservative replicate interval holds it from beta 1 up.

## Secondary outcomes

**Break-even lambda of the guideline against never measuring** ($ per undiagnosed condition-month):

| beta | base | W x0.5 | W x1.5 | lag x0.5 | lag x1.5 |
|---|---|---|---|---|---|
| 0 | 431 | 432 | 431 | 1,167 | 252 |
| 0.5 | 192 | 193 | 190 | 489 | 116 |
| 1 | 115 | 118 | 112 | 264 | 74 |
| 2 | 100 | 107 | 94 | 202 | 68 |

The clinical lag matters far more than the window: a shorter undiagnosed period makes measurement worth less,
since the clinic would have found the disease soon anyway. The window barely moves anything.

**Mean detection delay at beta 1** (months; 12-month 5.9 to 7.0, guideline 9.3 to 10.8, learned at $200 6.7 to
8.0 across settings): the ranking of policies is the same in every setting.

## Reading

- The simulator does not hand adaptive scheduling the win by construction. At beta 0 the learned policy loses
  at $100 and $200 in every setting except the 1.5x lag (the sanity failure above). At beta 0.25 and $200 it
  edges the best fixed policy by $8 to $50 in three settings and by $335 with the longer lag, but loses by
  $246 with the shorter lag, so no win there is robust, and none survives the replicate interval. The robust
  win appears once labs carry signal (beta 0.5 and up) and survives halving the onset window and the clinical
  lag or raising them by half.
- The answer is conditional on two things the simulator cannot settle: how informative real labs are (beta)
  and what a month of undiagnosed disease is worth (lambda). The MIMIC-IV lead-time check is the planned way
  to place real care on the beta axis.
- Hyperlipidemia's built-in benefit matters only at $100; above that the conclusion is the same without it.
- Limits: the learned policy was trained with hyperlipidemia in the reward and only re-scored without it;
  beta is a relative scale (pre-registered); three replicates make the replicate interval wide.
