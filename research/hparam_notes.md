# Hyperparameter search for Table 1 (ISEM 735, task 29)

Section 1 was written and committed **before any trial ran**. Results go in section 2 and
deviations in section 3, so a reader can check that nothing in section 1 moved after the numbers
came in.

## 1. Protocol (fixed before running)

**Why.** Table 1 (RQ1) compares XGBoost with the time-decay GRU and the visit-day Transformer at
hand-picked settings. A sequence model that loses at its defaults may only be under-tuned, so the
Part 2 draft promised an equal-size search for all three before RQ1 is final.

**Data and splits.** These are the leak-free features from `research/data.py` (default lab version) and the
same fit / validation / test split as tasks 7, 13 and 24 (`calibration_splits`, seed 42). The search
reads only the fit and validation splits. The test split is scored once, after the search, for
the five-seed refits. The validation split now does three jobs: early stopping (torch), model selection, and isotonic calibration (XGBoost). We accept that and do not split it again, so the test set stays
untouched and Table 1 stays comparable with task 24.

**Objective.** Minimise validation log loss of the raw (uncalibrated) model output. Log loss
penalises bad probabilities, not only bad rankings, which matters because the scheduler acts on
risk levels.

**Class weighting.** The GRU and Transformer train on unweighted BCE. The default XGBoost uses
`scale_pos_weight` (negatives / positives). Weighting inflates raw probabilities, which raw log
loss would punish even though isotonic calibration fixes it later. To score all three models
the same way, XGBoost is searched and refitted **unweighted** (`scale_pos_weight = 1`). The default
Table 1 XGBoost (weighted, then isotonic) stays in the table as the reference row.

**Budget and sampler.** 50 trials per model. Optuna 4.5 `TPESampler(seed=42)`. Trial 0 for each
model is its current default setting, enqueued, so the best configuration can never be worse than
the default on validation. Torch models use `MedianPruner(n_startup_trials=10, n_warmup_steps=5)`
on per-epoch validation loss. A pruned trial counts toward the 50. XGBoost trials take about a second each,
so they are not pruned. Torch trials train with seed 42, max 40 epochs and patience 4, the same as task 24.

**Search spaces** (each covers capacity, regularisation and learning rate):

| Model | Parameter | Range | Default (trial 0) |
|---|---|---|---|
| XGBoost | n_estimators | 100-1500, log | 300 |
| | max_depth | 2-8 | 3 |
| | learning_rate | 0.005-0.3, log | 0.05 |
| | subsample | 0.5-1.0 | 0.9 |
| | colsample_bytree | 0.3-1.0 | 0.9 |
| | min_child_weight | 1-50, log | 1 |
| | reg_lambda | 0.01-100, log | 1 |
| GRU | embed_dim | {16, 32, 64} | 32 |
| | hidden_dim | {32, 64, 128} | 64 |
| | dropout | 0.0-0.5 | 0.3 |
| | lr | 1e-4-1e-2, log | 1e-3 |
| | weight_decay | 1e-7-1e-3, log | 1e-5 |
| | batch_size | {128, 256, 512} | 256 |
| Transformer | dim | {32, 64, 128} | 64 |
| | heads | {2, 4} | 4 |
| | layers | {1, 2, 3} | 2 |
| | dropout | 0.0-0.5 | 0.3 |
| | lr | 1e-4-1e-2, log | 1e-3 |
| | weight_decay | 1e-7-1e-3, log | 1e-5 |
| | batch_size | {128, 256, 512} | 256 |

**Refits.** Each model's best configuration is retrained with seeds 42, 1, 2, 3 and 4 on the fit
split. As in Table 1, XGBoost is then isotonic-calibrated on validation and the torch models are scored raw. Test predictions go to
`data/processed/seed_predictions_tuned/`, apart from the task 24 files.

**Reported.** Tuned Table 1: AUROC and Brier with patient-level bootstrap CIs (2,000 resamples),
paired AUROC differences between models with bootstrap CI and DeLong p, the five-seed spread, and
each model's tuned minus default AUROC (seed 42, paired bootstrap).

**Decision rule for RQ1.** Let D = tuned XGBoost AUROC minus the better tuned sequence model's,
seed 42, 95% paired bootstrap CI.
- CI entirely above 0: RQ1's answer ("sequence models do not beat XGBoost") holds after tuning.
- CI spans 0: the gap was a tuning artifact. RQ1 is reported as "no difference".
- CI entirely below 0: the sequence model wins after tuning, and the report says so.

The number of seeds out of five where D > 0 is reported alongside as a robustness check.

**Scope.** Table 1 only (Chinmay, 2026-11-17). Tables 2 and 3 and the simulator predictors keep
the default XGBoost. If tuned XGBoost differs materially, the final report names that as a
limitation.

**Code.** `research/hparam_paths.py` holds the shared constants and imports neither library.
`research/hparam_torch.py` runs the GRU and Transformer search and refits (torch only).
`research/hparam_xgb.py` runs the XGBoost search and refits and builds the tuned Table 1 (XGBoost
only, run after the torch module).

## 2. Results

Run 2026-11-17. The torch search and refits took 25 min with 10 threads, and the XGBoost search took 2 min.
Trial 0 reproduced the task 24 GRU default exactly (validation log loss 0.4941).

**Search** (validation log loss, raw output; lower is better):

| Model | Default (trial 0) | Best | Best trial | Pruned | Best configuration |
|---|---|---|---|---|---|
| XGBoost (unweighted) | 0.4901 | 0.4866 | 25 | 0 / 50 | 1,166 trees, depth 6, lr 0.0068, subsample 0.95, colsample 0.43, min_child_weight 2.4, lambda 0.11 |
| GRU | 0.4941 | 0.4917 | 45 | 18 / 50 | embed 32, hidden 32, dropout 0.03, lr 0.0077, wd 5.7e-6, batch 128 |
| Transformer | 0.5062 | 0.4945 | 31 | 15 / 50 | dim 32, 2 heads, 1 layer, dropout 0.001, lr 0.0029, wd 8.1e-6, batch 256 |

Both sequence models ended up *smaller* than their defaults, with almost no dropout. The Transformer
shrank from 2 layers of width 64 to 1 layer of width 32. On about 25,000 fit patients the extra capacity was not
helping.

**The sequence models' optimum sits at the capacity floor of the space.** The Transformer's best setting
takes the smallest option for all three capacity parameters (dim 32, 2 heads, 1 layer). The GRU takes the
smallest hidden size (32) and the smallest batch size (128). Two more values are near an edge: GRU lr 0.0077 (upper
bound 0.01) and Transformer dropout 0.001 (lower bound 0). So the search may not have found the sequence
models' best settings. A smaller network might do better still. We did not widen the ranges, because that
would be a post-hoc change. A smaller-capacity run, if it is done, must be labelled supplementary.

**Tuned Table 1** (test split, seed 42, 95% patient-level bootstrap CI; `outputs/tables/table1_tuned_bootstrap_ci.csv`):

| Model | AUROC default | AUROC tuned | Tuned - default | Brier default | Brier tuned |
|---|---|---|---|---|---|
| XGBoost + isotonic | 0.8276 | 0.8286 [0.8210, 0.8361] | +0.0010 [-0.0006, 0.0026], DeLong p 0.23 | 0.1670 | 0.1661 |
| GRU | 0.8207 | 0.8222 [0.8142, 0.8296] | +0.0015 [-0.0003, 0.0032], p 0.09 | 0.1702 | 0.1694 |
| Transformer | 0.8128 | 0.8193 [0.8115, 0.8267] | +0.0065 [0.0037, 0.0091], p < 0.001 | 0.1743 | 0.1709 |

Paired differences between tuned models: XGBoost - GRU 0.0064 [0.0034, 0.0096] (DeLong p < 0.001);
XGBoost - Transformer 0.0094 [0.0058, 0.0128]; GRU - Transformer 0.0030 [0.0008, 0.0053].
Over five seeds (`table1_tuned_seeds.csv`) XGBoost - GRU ranges 0.0056 to 0.0080 and XGBoost - Transformer
0.0093 to 0.0107. Every seed has DeLong p < 0.001. Tuned XGBoost's seed SD is 0.0001.

**RQ1 decision (pre-registered rule).** The better tuned sequence model is the GRU. D = 0.0064 [0.0034,
0.0096], entirely above 0, and D > 0 on 5 of 5 seeds. **RQ1's answer holds after tuning:** neither sequence
model beats XGBoost. The task 24 gap (0.0069) barely moves, because tuning helped XGBoost and the GRU by
about the same small amount.

**What it means for the rest of the project.** Only the Transformer gained clearly from tuning (+0.0065). It
still ranks last, so its default was under-tuned but its rank was not an artifact. Tuning did not
significantly improve the default XGBoost (+0.0010, CI spans 0). The tuned row changes two things at once,
unweighted training and new settings. A supplementary refit (section 3) splits the gain into +0.0006 from
unweighting and +0.0004 from the settings, and neither is significant. On Table 1's data, then, keeping the default
XGBoost for Tables 2 and 3 costs about 0.001 AUROC. That does **not** carry over to the simulator predictors (tasks 17
onward). They use different features, they were never searched, and W7-2 found that noise lab columns hurt the
untuned model there. The final report should say they remain untuned.

## 3. Deviations from section 1

| Date | Change | Why |
|---|---|---|
| 2026-11-17 | The XGBoost search ran alongside the torch search, not after it. Its table step was re-run with `--skip-search` once the torch predictions existed. | Ordering only, to save time. Each search is seeded and reads the same splits, so the results are unaffected. |
| 2026-11-17 | Supplementary, not in section 1: default XGBoost settings refitted **unweighted** + isotonic, seed 42. Test AUROC 0.8282 (weighted default 0.8276, tuned 0.8286). Unweighted - weighted +0.0006 [-0.0003, 0.0015]; tuned - unweighted default +0.0004 [-0.0012, 0.0021]. | Separates the two changes inside the tuned XGBoost row. Scratch script, not in the repo. |
