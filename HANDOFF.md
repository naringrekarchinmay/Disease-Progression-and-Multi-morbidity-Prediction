# Project Handoff

## Summary

This is a healthcare machine learning course project that predicts disease
progression and multimorbidity risk from synthetic EHR data (2018–2024) for
100,000 patients.

Predictors use only events on or before **2021-12-31** (the cutoff date).
Targets are built from events between **2022-01-01** and **2024-12-31**:

- `future_hospitalization`
- `future_icu_admission`
- `future_30d_readmission`
- `multimorbidity_progression` (patients with < 2 baseline conditions who reach >= 2)

## How to Run

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Notebooks (full workflow with explanations)

Run `notebooks/01_data_loading.ipynb` through `notebooks/10_final_results_summary.ipynb`
in order. Each notebook states its purpose, inputs, and saved outputs.

### Pipeline runner (quick regeneration without notebooks)

```bash
python run_pipeline.py --step features   # rebuild data/processed/modeling_dataset.csv
python run_pipeline.py --step models     # train and save models to models/
python run_pipeline.py --step evaluate   # write metrics + comparison figures
python run_pipeline.py --step calibrate  # threshold table, calibration + PR curves
python run_pipeline.py --step explain    # SHAP summary + top-features table
python run_pipeline.py --step survival   # Kaplan-Meier + Cox progression model
python run_pipeline.py --step all        # everything above in order

# Faster runs on a random patient sample (useful for testing):
python run_pipeline.py --step all --sample 5000
```

The runner reuses the helpers in `src/` and does not replace the notebooks —
it is a convenience for regenerating key artifacts.

### Tests

```bash
pytest tests/
# or without pytest:
python tests/test_features.py && python tests/test_outputs.py
```

The tests focus on data-leakage prevention: target columns must never appear
in the model matrix, future information must stay out of predictors, and the
date cutoff must be respected.

### Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The dashboard has five tabs: **Overview** (dataset summary + target rates),
**Model Metrics** (best-per-target, all metrics, hospitalization threshold
table), **Figures** (grouped: EDA, model comparison, calibration, explainability,
survival), **Reports** (view the markdown report; download the DOCX/PDF), and
**Patient Risk Lookup** (pick a patient ID and see the best model's predicted
risk for both targets). It degrades gracefully if any artifact is missing.
Validated headlessly with Streamlit's `AppTest` — loads and runs with no
exceptions, including after selecting a patient.

## Current Metrics (test split)

All five models (dummy, logistic regression, random forest, XGBoost, LightGBM)
are trained for both targets. Best model per target:

| Target | Best model | ROC-AUC | F1 |
|---|---|---|---|
| future_hospitalization | XGBoost | 0.694 | 0.132 |
| multimorbidity_progression | XGBoost | 0.865 | 0.774 |

Notes: logistic regression is a strong hospitalization baseline (ROC-AUC 0.686);
random forest ties on multimorbidity (0.861). `future_hospitalization` is
predicted on all 100,000 patients; `multimorbidity_progression` is predicted
only within the at-risk cohort of 41,654 patients who start with fewer than two
baseline conditions (matching notebook 07). Both use a stratified 75/25 split.

Outcome rates in the 100,000-patient modeling dataset:

- Future hospitalization: 4.55%
- Future ICU admission: 0.10%
- Future 30-day readmission: 0.68%
- Multimorbidity progression: 18.11%

Full metric tables live in `outputs/tables/08_all_model_metrics.csv` and
`outputs/tables/08_best_models_by_target.csv`.

### Threshold and calibration (hospitalization)

`--step calibrate` analyses the best hospitalization model (XGBoost) and writes:

- `outputs/tables/threshold_analysis.csv` — precision, recall, F1, and the share
  of patients flagged at thresholds from 0.05 to 0.95.
- `outputs/figures/calibration_curve.png` — reliability diagram with Brier score.
- `outputs/figures/precision_recall_curve.png`.

Key finding: the class-weighted XGBoost ranks patients well (ROC-AUC 0.694) but
is **poorly calibrated** (Brier 0.211) — it inflates predicted risk, flagging
~93% of patients at a 0.10 threshold. Its scores are useful for *ranking* who to
review, not as literal probabilities. Recalibrating (e.g. `CalibratedClassifierCV`)
would be the next step before using absolute risk thresholds clinically.

### Explainability (SHAP)

`--step explain` runs SHAP `TreeExplainer` on the best tree model per target
(XGBoost for both), using a 500-row sample, and writes:

- `outputs/figures/shap_summary_<target>.png` — beeswarm summary plot.
- `outputs/tables/shap_top_features_<target>.csv` — features ranked by mean
  absolute SHAP value.

If `shap` is unavailable it falls back to plain feature importance
(`feature_importance_<target>.png/.csv`). Top drivers are clinically sensible:
prior hospitalization count, age, and Charlson index for hospitalization;
medication indication/count (polypharmacy), baseline condition count, and visit
count for multimorbidity progression.

### Survival analysis (time to multimorbidity progression)

`--step survival` studies how long the at-risk cohort (41,654 patients with
fewer than two baseline conditions) stays free of multimorbidity, with the event
being the day they reach two distinct conditions and censoring at the 2024-12-31
horizon. It writes:

- `outputs/figures/km_curve_progression.png` — Kaplan-Meier progression-free
  survival stratified by baseline condition count (0 vs 1).
- `outputs/tables/cox_model_summary.csv` — Cox proportional hazards summary
  (hazard ratios and confidence intervals).

Validation: the survival table's event counts (18,107 total) and per-group rates
exactly match the `multimorbidity_progression` classification target, confirming
the two are built from a consistent definition.

Findings: older age (HR ~1.02/year) and higher Charlson index (HR ~1.49) raise
the progression hazard, as expected. Counterintuitively, patients starting with
1 condition progress *slower* than those starting with 0 (HR ~0.37) — in this
synthetic data, 0-condition patients are more likely to accumulate two new
conditions than 1-condition patients are to add their second. This is a property
of the synthetic data, not a modeling error, and is worth noting in the report.

## Known Limitations

- The data is synthetic; results do not transfer to real patients.
- `future_hospitalization` is highly imbalanced (4.55% positive), and ICU /
  readmission targets are rarer still — accuracy is misleading for them; use
  ROC-AUC, average precision, recall, and F1.
- The saved `data/processed/modeling_dataset.csv` contains both predictors and
  target columns. Always build model matrices with
  `src.features.get_modeling_matrices`, which drops all target columns.
- XGBoost and LightGBM require the OpenMP runtime. If they fail to import with
  an OSError about `libomp.dylib`, install it with `brew install libomp`
  (already done on this machine).
- Saved `models/*.joblib` files only load with a scikit-learn version close to
  the one that trained them. If loading fails after an environment change,
  retrain with `python run_pipeline.py --step models`.
- SHAP and lifelines are listed in `requirements.txt` but not yet used in
  final outputs; they are optional and environment-dependent.
- The Streamlit app's patient risk lookup loads models on demand; the first
  lookup after startup is a little slow while models are cached.
- The report source of truth is `outputs/reports/final_project_report.md`. The
  `.docx` and `.pdf` are regenerated from it with pandoc (run from
  `outputs/reports/`):

  ```bash
  # DOCX (works with any pandoc, e.g. the anaconda one):
  pandoc final_project_report.md -o final_project_report.docx \
      --resource-path=.:.. --toc --toc-depth=2

  # PDF needs pandoc >= 3.1.7 + typst (both installed via Homebrew):
  /opt/homebrew/bin/pandoc final_project_report.md -o final_project_report.pdf \
      --pdf-engine=typst --resource-path=.:.. --toc --toc-depth=2
  ```

  Figure paths in the markdown are relative (`../figures/*.png`), so run pandoc
  from the reports directory with `--resource-path=.:..` so they embed.

## Repo Notes

- The folder is not yet a Git repository. `.gitignore` is ready if one is
  initialized; generated artifacts (models, figures, tables, processed CSVs)
  are ignored by default.
- Roadmap for further improvements is in `PROMPT.md` (phases 4–9: advanced
  models for hospitalization, calibration/threshold analysis, SHAP, survival
  analysis, app upgrade, report refresh).
