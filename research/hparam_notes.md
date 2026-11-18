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

(filled in after the runs)

## 3. Deviations from section 1

| Date | Change | Why |
|---|---|---|
