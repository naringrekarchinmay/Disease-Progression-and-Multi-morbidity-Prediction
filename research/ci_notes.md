# Confidence intervals and seed spread for Tables 1 and 2: notes

Written 2026-11-09 (task 24). Code: `research/stats_ci.py` (bootstrap and DeLong, tested in
`tests/test_stats_ci.py`), `research/seed_runs.py` (torch: GRU and Transformer over seeds),
`research/model_ci.py` (XGBoost seeds, both tables). Shared seed constants live in `research/seed_paths.py`,
which loads neither torch nor XGBoost (importing `seed_runs` from `model_ci` segfaulted on the libomp clash).
Tables: `outputs/tables/table1_bootstrap_ci.csv`, `table1_seeds.csv`, `table2_bootstrap_ci.csv`.
Prompted by Part 3 feedback, major issue 2.

## Method

- **Paired patient-level bootstrap**: 2,000 resamples of the 10,414 test patients (seed 42); every model or
  schedule is scored on the same resample, so differences are paired. 95% percentile intervals.
- **DeLong test** for paired AUROC differences (DeLong et al., 1988), in the fast form of Sun and Xu (2014).
  A test checks that its standard error agrees with the bootstrap's to within 20%; on Table 1 they agree to
  the fourth decimal.
- **Seeds**: XGBoost, GRU and Transformer each retrained with seeds 42, 1, 2, 3, 4 on the same fit, validation
  and test split, so only initialisation, dropout, subsampling and batch order change. Sequence models are
  scored uncalibrated, XGBoost with its isotonic step, as in Table 1.
- The Table 2 bootstrap resamples patients for a fixed fitted model; it does not include refitting variance.
  For Table 1 that variance is the seed spread, which is small (below).

## Table 1 (RQ1), seed 42

| Model | AUROC (95% CI) | Brier (95% CI) |
|---|---|---|
| Calibrated XGBoost | 0.828 (0.820 to 0.835) | 0.167 (0.163 to 0.171) |
| GRU | 0.821 (0.813 to 0.828) | 0.170 (0.167 to 0.174) |
| Transformer | 0.813 (0.805 to 0.821) | 0.174 (0.171 to 0.178) |

| Paired AUROC difference | Estimate (95% CI) | DeLong p |
|---|---|---|
| XGBoost - GRU | 0.0069 (0.0039 to 0.0100) | 1.9e-5 |
| XGBoost - Transformer | 0.0148 (0.0111 to 0.0183) | 8.6e-16 |
| GRU - Transformer | 0.0079 (0.0055 to 0.0103) | 8.2e-11 |

The separate CIs overlap, but the paired ones do not cross zero: the models are scored on the same patients,
and most of each CI's width is which patients happen to be in the test set, which cancels in a difference.

## Table 1 over five seeds

| | Mean AUROC | SD | Range |
|---|---|---|---|
| XGBoost | 0.8283 | 0.0004 | 0.8276 to 0.8288 |
| GRU | 0.8203 | 0.0007 | 0.8194 to 0.8209 |
| Transformer | 0.8132 | 0.0013 | 0.8114 to 0.8146 |
| XGBoost - GRU | 0.0080 | 0.0008 | 0.0069 to 0.0089 |
| XGBoost - Transformer | 0.0151 | 0.0010 | 0.0141 to 0.0168 |

XGBoost beats both sequence models at every seed, with DeLong p at most 1.9e-5. **RQ1's answer stands and is
now shown**: the gap is small (under 0.01 against the GRU) but real, and seed choice moves AUROC by about a
tenth of it. The GRU at seed 42 reproduces task 7 exactly (0.8207); the Transformer gives 0.8128 against 0.8135
in task 13, so its training is not bit-for-bit reproducible on this machine. The spread over seeds covers it.

## Table 2 (RQ2 under masking), paired AUROC differences

Random = the mean of the five budget-matched random schedules, averaged inside each resample.

| Comparison | All tests (original) | Universal panel | Panel + W |
|---|---|---|---|
| Keep all - keep none (labs' value) | 0.0104 (0.0074 to 0.0135) | 0.0099 (0.0070 to 0.0128) | 0.0109 (0.0082 to 0.0137) |
| 12-month - keep all | 0.0005 (0.0000 to 0.0011) | -0.0003 (-0.0008 to 0.0003) | -0.0002 (-0.0008 to 0.0004) |
| Threshold tau 0.1 - random | 0.0006 (-0.0012 to 0.0024) | 0.0000 (-0.0013 to 0.0015) | 0.0008 (-0.0006 to 0.0022) |
| Threshold tau 0.2 - random | 0.0022 (0.0005 to 0.0039) | 0.0025 (0.0008 to 0.0041) | 0.0016 (0.0001 to 0.0032) |
| Threshold tau 0.3 - random | 0.0009 (-0.0001 to 0.0019) | 0.0013 (0.0004 to 0.0023) | 0.0007 (-0.0004 to 0.0017) |

All seven tau values are in the table file.

**Reading.**
- Labs are worth about 0.01 AUROC in every version, clearly above zero.
- A fixed 12-month schedule scores the same as keeping every lab: the CI includes zero once the artifacts are
  removed, and is at most +0.001 with them.
- The threshold rule's edge over budget-matched random visits is real at some budgets (tau 0.2 in all three
  versions) and indistinguishable from zero at others. The October claim "the rule beats random slightly" is
  right only in that weak, budget-dependent sense, and the report should say so with the intervals.
- Nothing here changes a conclusion; it shows which differences can be read.

## References

DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated
receiver operating characteristic curves: A nonparametric approach. *Biometrics, 44*(3), 837-845.
https://doi.org/10.2307/2531595

Sun, X., & Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated
receiver operating characteristic curves. *IEEE Signal Processing Letters, 21*(11), 1389-1393.
https://doi.org/10.1109/LSP.2014.2337313
