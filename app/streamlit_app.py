"""Streamlit dashboard for the disease progression ML project.

Run with:  streamlit run app/streamlit_app.py
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features import TARGET_COLUMNS, get_modeling_matrices
from src.modeling import load_model

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

PRIMARY_TARGETS = ["future_hospitalization", "multimorbidity_progression"]

# Group figures into readable sections by filename keyword.
FIGURE_SECTIONS = {
    "EDA": ("01_", "02_", "03_", "04_", "05_"),
    "Model comparison": ("06_", "07_", "08_", "roc_curves"),
    "Calibration & thresholds": ("calibration", "precision_recall"),
    "Explainability": ("shap", "feature_importance", "09_"),
    "Survival": ("km_",),
}


st.set_page_config(page_title="Disease Progression ML", page_icon="🩺", layout="wide")


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_cached_model(path_str: str):
    return load_model(path_str)


def figure_section(name: str) -> list[Path]:
    prefixes = FIGURE_SECTIONS[name]
    return sorted(p for p in FIGURES_DIR.glob("*.png") if any(k in p.name for k in prefixes))


st.title("Disease Progression and Multimorbidity Prediction")
st.write(
    "Synthetic EHR project using baseline patient history before 2022 to predict "
    "future hospitalization and multimorbidity progression through 2024."
)

if not DATA_PATH.exists():
    st.warning("Run `python run_pipeline.py --step features` first to create the modeling dataset.")
    st.stop()

feature_df = load_csv(DATA_PATH)

overview_tab, metrics_tab, figures_tab, reports_tab, lookup_tab = st.tabs(
    ["Overview", "Model Metrics", "Figures", "Reports", "Patient Risk Lookup"]
)


with overview_tab:
    st.subheader("Modeling Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients", f"{len(feature_df):,}")
    col2.metric("Features", f"{feature_df.shape[1]:,}")
    col3.metric("Future Hospitalization", f"{feature_df['future_hospitalization'].mean():.1%}")
    col4.metric("Progression Risk", f"{feature_df['multimorbidity_progression'].mean():.1%}")

    st.caption("First 25 rows of the modeling dataset")
    st.dataframe(feature_df.head(25), width="stretch")

    st.subheader("Target Rates")
    target_cols = [c for c in TARGET_COLUMNS if feature_df[c].dropna().isin([0, 1]).all()]
    target_rates = feature_df[target_cols].mean().sort_values(ascending=False).rename("rate")
    st.bar_chart(target_rates)


with metrics_tab:
    best_path = TABLES_DIR / "08_best_models_by_target.csv"
    if best_path.exists():
        st.subheader("Best Model per Target")
        st.dataframe(load_csv(best_path), width="stretch")

    metrics_path = TABLES_DIR / "08_all_model_metrics.csv"
    if metrics_path.exists():
        st.subheader("All Model Metrics")
        st.dataframe(load_csv(metrics_path), width="stretch")
    else:
        st.info("Run `python run_pipeline.py --step evaluate` to populate model metrics.")

    threshold_path = TABLES_DIR / "threshold_analysis.csv"
    if threshold_path.exists():
        st.subheader("Hospitalization Threshold Trade-offs")
        st.caption("Precision, recall, and the share of patients flagged at each decision threshold.")
        st.dataframe(load_csv(threshold_path), width="stretch")


with figures_tab:
    available = {name: figure_section(name) for name in FIGURE_SECTIONS}
    available = {name: figs for name, figs in available.items() if figs}
    if not available:
        st.info("Run the modeling and analysis steps to generate figures.")
    else:
        section = st.radio("Figure group", list(available), horizontal=True)
        for fig_path in available[section]:
            st.image(str(fig_path), caption=fig_path.name, width="stretch")


with reports_tab:
    st.subheader("Project Reports")
    report_files = sorted(REPORTS_DIR.glob("*")) if REPORTS_DIR.exists() else []
    if not report_files:
        st.info("No reports found in outputs/reports/.")
    else:
        for report in report_files:
            st.markdown(f"**{report.name}**")
            if report.suffix == ".md":
                with st.expander(f"View {report.name}"):
                    st.markdown(report.read_text())
            else:
                st.download_button(
                    f"Download {report.name}",
                    data=report.read_bytes(),
                    file_name=report.name,
                )


with lookup_tab:
    st.subheader("Single-Patient Risk Lookup")
    best_path = TABLES_DIR / "08_best_models_by_target.csv"
    if not best_path.exists():
        st.info("Run `python run_pipeline.py --step evaluate` to choose the best models first.")
    else:
        best_models = load_csv(best_path).set_index("target")["model"].to_dict()

        patient_id = st.selectbox("Patient ID", feature_df["patient_id"].astype(str).tolist())
        patient_pos = feature_df.index[feature_df["patient_id"].astype(str) == patient_id][0]

        st.caption("Predicted risk from the best model for each target")
        cols = st.columns(len(PRIMARY_TARGETS))
        for col, target in zip(cols, PRIMARY_TARGETS):
            model_name = best_models.get(target)
            model_path = MODELS_DIR / f"{target}_{model_name}.joblib"
            if model_name is None or not model_path.exists():
                col.warning(f"No saved model for {target}.")
                continue
            X, _ = get_modeling_matrices(feature_df, target)
            model = load_cached_model(str(model_path))
            risk = float(model.predict_proba(X.iloc[[patient_pos]])[0, 1])
            col.metric(f"{target} ({model_name})", f"{risk:.1%}")

        baseline_count = feature_df.loc[patient_pos, "baseline_condition_count"]
        if baseline_count >= 2:
            st.caption(
                "Note: multimorbidity progression is defined only for patients with fewer than "
                "two baseline conditions, so its risk is not meaningful for this patient "
                f"(baseline condition count = {int(baseline_count)})."
            )

        with st.expander("Show this patient's feature values"):
            patient_features = feature_df.loc[[patient_pos]].T.reset_index()
            patient_features.columns = ["feature", "value"]
            patient_features["value"] = patient_features["value"].astype(str)
            st.dataframe(patient_features, width="stretch", hide_index=True)
