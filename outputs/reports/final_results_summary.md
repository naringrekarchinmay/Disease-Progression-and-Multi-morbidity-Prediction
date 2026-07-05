# Disease Progression and Multimorbidity Trajectory Prediction

## Situation
Healthcare teams need earlier signals of patients who may progress into multimorbidity or need future hospital care.

## Task
Use synthetic EHR data from 2018 to 2024 to build leakage-aware machine learning models for progression and risk prediction.

## Action
The project loads and cleans patient, diagnosis, lab, medication, and outcome tables; engineers baseline features before 2022; trains baseline and advanced models (logistic regression, random forest, XGBoost, LightGBM); evaluates multiple metrics; calibrates risk and analyzes decision thresholds; explains model drivers with SHAP; and models time to progression with Kaplan-Meier and Cox survival analysis.

## Results
- Best model for future_hospitalization: XGBoost (ROC-AUC=0.694, F1=0.132, Recall=0.677); logistic regression a close baseline (ROC-AUC=0.686).
- Best model for multimorbidity_progression: XGBoost (ROC-AUC=0.865, F1=0.774, Recall=0.869) on the at-risk cohort; random forest and LightGBM within 0.004 ROC-AUC.
- The best hospitalization model ranks well but is miscalibrated (Brier=0.211), so scores are best used for ranking, not as absolute risks.
- Survival analysis: older age (HR~1.02/yr) and higher Charlson index (HR~1.49) raise progression hazard.

## Future Work
Recalibrate predicted risk probabilities and tune operating thresholds, model the rarer ICU and readmission targets, and validate the workflow on another synthetic or de-identified EHR dataset.