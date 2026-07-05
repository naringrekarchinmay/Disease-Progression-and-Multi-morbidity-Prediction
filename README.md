# Disease Progression and Multimorbidity Trajectory Prediction

This project uses synthetic electronic health record (EHR) data from 2018 to 2024 to predict disease progression and multimorbidity risk at the patient level.

The project is structured for an ANLY 530 style machine learning course project: it includes problem definition, data loading, cleaning, exploratory analysis, feature engineering, baseline models, advanced models, evaluation, explainability, and a final results summary.

## Project Question

Can patient demographics, baseline diagnoses, lab history, medication history, and prior hospitalization patterns predict:

- Future hospitalization risk from 2022 through 2024
- Future ICU admission risk
- Future 30-day readmission risk
- Progression into multimorbidity among patients with fewer than two baseline conditions

## Data

Raw synthetic EHR files are expected in `data/raw/`. This repository currently also supports the CSV files directly under `data/`, which matches the provided starting dataset.

Expected source files:

- `patients.csv`
- `diagnoses.csv`
- `lab_results.csv`
- `medications.csv`
- `outcomes.csv`

## Leakage-Aware Design

The modeling notebooks use a time cutoff:

- Predictors: events on or before `2021-12-31`
- Targets: future events from `2022-01-01` through `2024-12-31`

The patient-level `dx_*` columns are excluded from model predictors because they may represent all-time diagnoses. Instead, baseline condition features are derived from dated diagnosis events before the cutoff.

## Repository Structure

```text
disease-progression-multimorbidity/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
├── models/
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── reports/
├── app/
├── requirements.txt
├── README.md
├── .gitignore
└── AGENTS.md
```

## Notebook Workflow

Run the notebooks in order:

1. `01_data_loading.ipynb`
2. `02_data_cleaning.ipynb`
3. `03_eda_patient_profiles.ipynb`
4. `04_multimorbidity_patterns.ipynb`
5. `05_feature_engineering.ipynb`
6. `06_baseline_models.ipynb`
7. `07_progression_prediction_models.ipynb`
8. `08_model_evaluation.ipynb`
9. `09_explainability_shap.ipynb`
10. `10_final_results_summary.ipynb`

Each notebook includes purpose, imports, data loading, main analysis or modeling, key findings, and saved outputs.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Optional libraries such as XGBoost, LightGBM, SHAP, and lifelines are included in `requirements.txt` because they are useful for advanced modeling and explainability.

## Outputs

Generated artifacts are written to:

- `data/processed/modeling_dataset.csv`
- `outputs/figures/`
- `outputs/tables/`
- `outputs/reports/final_results_summary.md`
- `models/`

## Streamlit App

After running the modeling notebooks, launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

The app summarizes the processed dataset, model metrics, and available figures.

## Course Guideline Coverage

This project directly addresses the requested sections:

- Problem definition and motivation
- EDA and pre-processing
- Feature definition and refinement
- Baseline and advanced modeling
- Train/test validation
- Multiple performance metrics
- Model comparison tables and graphs
- Explainability through feature importance and optional SHAP
- Final summary suitable for report and presentation preparation

