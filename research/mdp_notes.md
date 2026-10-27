# Week 9, part 1: the decision process and true policy values (W9-1, task 19)

Written 2026-10-26. Code: `research/simulator_mdp.py` (plus a stochastic-policy hook in
`research/simulator_policy.py`). Tests: `tests/test_simulator_mdp.py`. Table:
`outputs/tables/simulator_true_values.csv`. Logged data: `data/processed/simulated/logged/` (gitignored,
418 MB, 2.8 million patient-quarters per beta).

## Definition (confirmed by Chinmay, 2026-10-26)

- **State**: what the quarterly policy loop shows a policy (covariates, diagnoses so far, the latest and
  previous trigger-test values, months since the last lab, calendar month).
- **Action**: measure (full panel) or not, once a quarter, 28 quarters over 2018-2024.
- **Reward per quarter**: -$214.49 per measurement - lambda per month in which a lab-linked condition has
  truly begun but is undiagnosed (prevalent onsets count from January 2018). Lambda is swept over $25, $50,
  $100 and $200 per condition-month, with $50 as the planned headline. It is a judgement: no published
  price for a month of delayed detection fits this setting.
- **Discount**: 3% a year, the Second Panel on Cost-Effectiveness's reference-case rate (Sanders et al.,
  2016), so 1.03^(-1/4) = 0.9926 per quarter. The primary JAMA text sits behind a bot check, so the 3%
  figure was confirmed through secondary sources (cost-effectiveness papers citing the panel), not the
  paper itself.
- **True value** of a policy: mean discounted return over the seed-42 evaluation cohort (100,000 patients,
  the same truth and common random numbers as tasks 16 to 18), computed exactly from the simulator.
- **Behavior policy** for the logged data: follow a quarterly guideline rule with probability 0.8,
  otherwise measure with probability 0.25. The recorded propensity is 0.85 when the guideline says measure
  and 0.05 otherwise. Logged measure rate: 15.4% to 16.0% of quarters (0.62 to 0.64 a year).

Every value is linear in lambda: V = -C - lambda x U, where C is the discounted measurement cost and U the
discounted undiagnosed condition-months per patient. That gives each policy's **break-even lambda against
never measuring**, C / (U_never - U), above which the policy is worth its cost.

## Results

True value per patient at lambda = $50 / $200 (beta 1), and the break-even lambda against never measuring.

| Policy | Measurements/yr | Value, $50 | Value, $200 | Break-even, beta 0 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|---|---|
| Never measure | 0 | **-709** | -2,834 | | | | |
| Observed care | 0.37 | -1,014 | -2,554 | $431 | $199 | $128 | $121 |
| Quarterly guideline | 0.6 | -1,151 | **-2,246** | $426 | $190 | $114 | $100 |
| Behavior policy | 0.64 | -1,222 | -2,268 | $431 | $195 | $121 | $106 |
| 12-month | 1 | -1,632 | -2,443 | $520 | $235 | $155 | $135 |
| 6-month | 2 | -2,905 | -3,454 | $791 | $354 | $259 | $226 |
| 3-month | 4 | -5,560 | -5,904 | $1,352 | $588 | $459 | $412 |

Never measuring leaves 14.2 discounted undiagnosed condition-months per patient. The 12-month schedule cuts
that to 5.4 at beta 1 for $1,361 of discounted measurement cost; the guideline cuts it to 7.3 for $786.

## Reading

1. **At the planned headline lambda of $50, never measuring is the best policy at every beta.** A
   measurement costs $214.49, and at beta 1 even the most cost-efficient schedule (the guideline) saves
   only about 6.9 condition-months for $786, roughly $114 per month saved. A learned policy at $50 would
   rightly learn to measure almost nobody, which makes the week 9 comparison degenerate.
2. **Measuring pays only above about $100 to $155 per undiagnosed condition-month at beta 1 and 2**, and
   above $190 at beta 0.5. At beta 0 and 0.25 even the cheapest schedule needs $330 or more, because labs
   then find disease only through the glucose artifact.
3. **At $200 the guideline is the best of these policies at beta 0.5, 1 and 2**, ahead of observed care and
   12-month. That is the regime where a learned policy has room to beat it.
4. The behavior policy's value is within 1% to 6% of the guideline's at every lambda, as intended: the
   logged data looks like guideline care with some exploration.

## Decision (2026-10-26)

Chinmay moved the headline lambda to $200 and extended the sweep to $400. At $50 the finding stands as
reported above: measurement is not worth its cost below about $115 per condition-month, even at beta 1.

## Reference

Sanders, G. D., Neumann, P. J., Basu, A., Brock, D. W., Feeny, D., Krahn, M., Kuntz, K. M., Meltzer, D. O.,
Owens, D. K., Prosser, L. A., Salomon, J. A., Sculpher, M. J., Trikalinos, T. A., Russell, L. B., Siegel,
J. E., & Ganiats, T. G. (2016). Recommendations for conduct, methodological practices, and reporting of
cost-effectiveness analyses: Second Panel on Cost-Effectiveness in Health and Medicine. *JAMA, 316*(10),
1093-1103. https://doi.org/10.1001/jama.2016.12195

---

# Week 9, part 2: learned policy by fitted Q-iteration (W9-2, task 20)

Added 2026-10-26. Code: `research/simulator_fqi.py`. Tests: `tests/test_simulator_fqi.py`. Table:
`outputs/tables/simulator_fqi_policy.csv` (4 minutes).

## Method

- **Offline learning, as from real records.** The policy is learned only from logged data: 30,000 patients
  (seed 43) cared for by the behavior policy (guideline with 20% exploration), 840,000 patient-quarters per
  beta. It never queries the simulator while learning.
- **Fitted Q-iteration by backward induction.** The horizon is 28 quarters and the month is in the state, so
  for quarter t the target is r_t + gamma x max_a Q_{t+1}(s', a), regressed on the state with one XGBoost
  per action (150 trees, depth 4, minimum child weight 20). The policy measures when Q(measure) >
  Q(wait).
- **True value** comes from running the greedy policy on the seed-42 evaluation cohort (100,000 patients) and
  scoring it exactly, next to the policies in part 1.
- **Oracle**: knows every onset and measures once at the first quarter after it. No real policy can
  reach it; it shows how much room there is.
- Learned at lambda $100, $200 (headline) and $400.

## Results

True value per patient (higher is better), at the headline lambda of $200:

| beta | Learned policy | Best fixed policy | Gain | Oracle | Share of gap to oracle closed |
|---|---|---|---|---|---|
| 0 | -2,814 | never measure, -2,834 | +20 | -2,495 | 6% |
| 0.25 | -2,786 | never measure, -2,834 | +48 | -2,341 | 10% |
| 0.5 | -2,145 | guideline, -2,795 | +650 | -1,780 | 64% |
| 1 | -1,669 | guideline, -2,246 | +577 | -813 | 40% |
| 2 | -1,238 | guideline, -2,046 | +808 | -337 | 47% |

At beta 1 and $200 the learned policy measures 0.52 times a year (guideline 0.58) and cuts the restricted
mean delay to 7.5 months (guideline 10.1, 12-month 6.5). At $400 it beats the 12-month schedule, the best
fixed policy there, on both counts at once: 0.77 measurements a year against 1.0, and 5.6 months against
6.5, for a value of -2,506 against -3,524.

Other lambdas: at $400 the learned policy beats the best fixed policy at every beta (+324 to +1,369). At
$100 it beats it from beta 0.5 up (+137 to +478) but loses at beta 0 and 0.25 (-40 and -107), where never
measuring is best.

**FQI overestimates its own value.** Its estimate of the starting value is higher than the truth in every
one of the 15 settings, by $114 to $1,415 per patient (beta 1, $200: estimated -1,403, true -1,669). Taking
the maximum over noisy Q estimates inflates them; week 9's off-policy evaluation has to correct this.

**Who it measures**: under-50s at 0.53 to 0.87 of a fair share, over-80s at 1.07 to 1.25. That follows risk
(onsets are rarer in the young), but it is the same direction as the closeness rule in part 3, so RQ4 has to
report it.

## Reading

1. **A learned policy beats every fixed schedule and the guideline once labs carry signal** (beta 0.5 and
   up), at the headline price and above. The one-step threshold rules of task 18 could not. The difference
   is that fitted Q-iteration values a measurement by what it is expected to save, so it measures the
   patients whose labs are rising and whose risk is high. The closeness rule did the opposite.
2. It closes 40% to 64% of the gap between the best fixed policy and the oracle at beta 0.5 to 2 ($200).
   The rest is the price of not knowing onset.
3. Where labs are uninformative (beta 0 and 0.25), the learned policy can only match never measuring, and
   at $100 it is slightly worse: optimistic Q values make it measure when it should not.

## Limits

- **The reward uses truth.** Undiagnosed condition-months need the true onset, which real records do not
  have. In the simulator that is fine for learning and checking; with real data the reward would need a
  proxy (for example, months from a later diagnosis back to the first abnormal value).
- One seed, one training cohort, default regressor settings, no confidence intervals yet.
- The learned policy is evaluated in the same simulator it was trained on. That is in-distribution: it shows
  what is achievable when the model of disease is right, not robustness to a wrong one.

---

# Week 9, part 3: off-policy evaluation against the truth (W9-3, task 21)

Added 2026-10-26. Code: `research/simulator_ope.py`. Tests: `tests/test_simulator_ope.py`. Table:
`outputs/tables/simulator_ope.csv` (about 30 minutes).

## Method

Every estimate uses only the seed-42 logged dataset: 100,000 patients x 28 quarters under the behavior policy,
with recorded propensities (0.85 or 0.05). Each is compared with the target's true value on the same patients.

- **FQE**, the proposal's committed estimator (Le et al., 2019): backward induction on the logged rows with
  the target's action in the next state, one XGBoost per action and quarter (80 trees, depth 4).
- **WPDIS**: per-decision importance weights, normalized across patients at each quarter (Thomas &
  Brunskill, 2016).
- **WDR**: weighted doubly robust (Thomas & Brunskill, 2016, building on Jiang & Li, 2016), with FQE as the
  model.
- 95% percentile intervals for WPDIS and WDR from 200 patient bootstraps. FQE is not bootstrapped.
- Targets: never measure, the quarterly guideline, the behavior policy itself (an on-policy check), and the
  task 20 learned policies at $200 and $400, retrained identically (their true values reproduce task 20).
  This gives 50 estimates per estimator: 5 targets x 5 betas x 2 lambdas for the fixed targets, and each
  learned policy at its own lambda.

## Results

Median absolute relative error (largest in brackets):

| Target | Support in the logs (final-quarter effective sample size) | FQE | WPDIS | WDR | FQI's own estimate |
|---|---|---|---|---|---|
| Behavior policy (on-policy) | 100,000 | 0.0% | 0.0% | 0.0% | |
| Quarterly guideline | about 15,000 | 0.4% (0.7%) | 0.0% (0.4%) | 0.2% (0.5%) | |
| Learned policies | 0 to 4 | 3.6% (6.5%) | 19% (46%) | 16% (36%) | 19% (28%) |
| Never measure | 0 | 4.9% (76%) | 57% | 24% (29%) | |

Bootstrap intervals cover the truth for the guideline and the on-policy check every time. For the learned
policies they cover it 70% (WPDIS) and 90% (WDR) of the time; for never measure 0% and 40%.

**Choosing the best policy.** In all 10 beta-lambda settings the true best policy is the learned one. FQE
picks it 10 times out of 10, WDR 7 and WPDIS 4. Over the full ordering of never measure, guideline and
learned policy, FQE and WDR are right 7 times out of 10 and WPDIS never. FQE's misorderings are all
between never measure and the guideline, in settings where their true values are within 1.5% of each
other or where FQE's never-measure estimate fails (below).

**FQE's failure.** For never measure at beta 1 and 2, FQE is off by 55% and 76% (beta 1, $200: estimated
-4,397 against a true -2,834). The behavior policy measures almost everyone eventually, so the logs hold
almost no late-quarter states of a patient who has never had a lab. FQE has to extrapolate there, and it
extrapolates worse when labs matter more. At beta 0 to 0.5 the same target is within 5%.

## Reading

1. **FQE works where the proposal needs it.** For the learned policies, where the importance weights collapse
   (effective sample size 0 to 4 out of 100,000), FQE is within 3.6% (median) of the truth, and it always
   identifies the best policy. It is also far more accurate than the learned policy's own value estimate
   (19% median error, always optimistic). The proposal's choice of FQE as the main estimator holds up, and the
   pre-registered fallback is not needed for these targets.
2. **Importance sampling fails for deterministic policies over 28 decisions.** A trajectory keeps weight only
   if every one of its 28 logged actions matches the target. For the learned policies only 0 to 27 of 100,000
   do. WPDIS then has nothing to average at late quarters, so it is biased, and its intervals do not reflect it.
   WDR repairs part of this through the model, but not enough to trust it.
3. **FQE is only as good as the logs' coverage.** It fails exactly where the target visits states the
   behavior policy almost never produces (never measuring, when labs are informative). Coverage, not the
   estimator, is the binding constraint. This is the support problem found with the Kaggle data in week 2,
   now measured.
4. All three estimators are exact on-policy, which checks the code.

## Limits

- One logged dataset and one behavior policy; FQE has no interval.
- The truth is the simulator's. FQE was tested on data drawn from the same model it then estimates, so this
  checks the estimator, not the simulator.

## References

Ernst, D., Geurts, P., & Wehenkel, L. (2005). Tree-based batch mode reinforcement learning. *Journal of Machine
Learning Research, 6*, 503-556.

Jiang, N., & Li, L. (2016). Doubly robust off-policy value evaluation for reinforcement learning. In
*Proceedings of the 33rd International Conference on Machine Learning* (Vol. 48, pp. 652-661). PMLR.

Le, H., Voloshin, C., & Yue, Y. (2019). Batch policy learning under constraints. In *Proceedings of the 36th
International Conference on Machine Learning* (Vol. 97, pp. 3703-3712). PMLR.

Thomas, P., & Brunskill, E. (2016). Data-efficient off-policy policy evaluation for reinforcement learning. In
*Proceedings of the 33rd International Conference on Machine Learning* (Vol. 48, pp. 2139-2148). PMLR.

(Ernst et al. is the fitted Q-iteration reference for part 2.)
