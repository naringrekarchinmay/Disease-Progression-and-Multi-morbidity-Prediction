# Generator leak in the medication table: notes

Found 2026-10-12 while calibrating the simulator. Code: `research/data.py`. Tests: `tests/test_data.py`.

## What the Kaggle generator does

- **Medications encode all-time diagnoses.** For every condition first diagnosed after the 2021-12-31
  cutoff (52,083 patient-condition pairs), the medication for it had already started before the
  cutoff: 100% of them, a median of 742 days before the diagnosis. Another 18.6% of medication
  starts are for conditions the patient is never diagnosed with.
- **Labs follow all-time diagnosis flags, not timing.** Patients ever diagnosed with chronic kidney
  disease have mean creatinine 5.74 against 0.90; ever-hyperlipidemic patients have LDL 142 against 64.
  Pre-cutoff creatinine predicts a future kidney disease diagnosis with AUROC 0.982, and pre-cutoff
  glucose predicts future diabetes with AUROC 0.72. BNP shows no relationship to heart failure.

Our pipeline only ever used events before the cutoff, so this is not leakage in our code. The generator
writes future diagnoses into past events, and a model that reads medications can predict a "new"
condition by finding its drug.

## Fix

A medication cannot start before its condition's first diagnosis: earlier starts move to the diagnosis
date, and medications for conditions never diagnosed are dropped (296,506 medication rows kept). Labs
are left as they are, because labs that show a disease before it is diagnosed are the signal measurement
scheduling relies on. Targets are unchanged (checked in the tests). The original pipeline, its saved
models and the Task 3 audit notebook still describe the original data; the research pipeline reads
`modeling_dataset_leakfree.csv` and `sequences_leakfree.parquet`.

## What changed (multimorbidity progression, test set, seed 42)

| Result | Before | Leak-free |
|---|---|---|
| Calibrated XGBoost AUROC / Brier | 0.864 / 0.144 | 0.828 / 0.167 |
| GRU AUROC / Brier / calibration slope | 0.854 / 0.150 / 0.99 | 0.821 / 0.170 / 1.00 |
| Transformer AUROC (learning rate chosen on validation) | 0.845 (3e-4) | 0.814 (1e-3) |
| Masking spread, keep none to keep all | 0.0009 (flat) | 0.010 (0.817 to 0.828) |
| GRU risk change when the last lab visit is hidden, median | 0.013 | 0.026 |
| AUROC with the last lab visit hidden | 0.853 | 0.802 |

Statements made before this fix that labs "carry about 4% of the signal" or are "uninformative" were
wrong: medications carried the same information, so removing labs cost nothing. With medications fixed,
labs matter. The proposal's week 5 rule still applies, since both sequence models still trail the
calibrated XGBoost, though the GRU's gap narrowed from 0.010 to 0.007 AUROC.

The Charlson index is also a single undated value per patient that tracks future conditions; dropping it
cost 0.001 AUROC in Task 7, so it was left in and noted as a caveat.
