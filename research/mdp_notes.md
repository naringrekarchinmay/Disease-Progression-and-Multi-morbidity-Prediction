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

## Implication for W9-2 (needs a decision)

Lambda is a judgement, and the headline value decides whether the learning problem is interesting.
Options: move the headline to $200 and extend the sweep upward (for example $400), so the learned policy
is tested where measurement can pay; or keep $50 and report "measurement is not worth its cost below about
$115 per condition-month" as the finding, with the learned policy run at $200 only.

## Reference

Sanders, G. D., Neumann, P. J., Basu, A., Brock, D. W., Feeny, D., Krahn, M., Kuntz, K. M., Meltzer, D. O.,
Owens, D. K., Prosser, L. A., Salomon, J. A., Sculpher, M. J., Trikalinos, T. A., Russell, L. B., Siegel,
J. E., & Ganiats, T. G. (2016). Recommendations for conduct, methodological practices, and reporting of
cost-effectiveness analyses: Second Panel on Cost-Effectiveness in Health and Medicine. *JAMA, 316*(10),
1093-1103. https://doi.org/10.1001/jama.2016.12195
