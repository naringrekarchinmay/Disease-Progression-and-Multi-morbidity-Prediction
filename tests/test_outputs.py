"""Checks that the key generated project artifacts exist and look sane.

Run with `pytest tests/` or directly with `python tests/test_outputs.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import TARGET_COLUMNS

MODELING_DATASET = PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"
METRICS_TABLE = PROJECT_ROOT / "outputs" / "tables" / "08_all_model_metrics.csv"


def test_modeling_dataset_exists_with_targets():
    assert MODELING_DATASET.exists(), f"Missing {MODELING_DATASET}; run the pipeline features step."
    columns = pd.read_csv(MODELING_DATASET, nrows=0).columns
    assert "patient_id" in columns
    for target in TARGET_COLUMNS:
        assert target in columns, f"Missing target column in modeling dataset: {target}"


def test_saved_model_metrics_exist():
    assert METRICS_TABLE.exists(), f"Missing {METRICS_TABLE}; run the evaluation notebook or pipeline."
    metrics = pd.read_csv(METRICS_TABLE)
    for col in ["model", "target", "roc_auc", "f1"]:
        assert col in metrics.columns, f"Missing metrics column: {col}"
    assert len(metrics) > 0


def test_saved_models_exist():
    models = list((PROJECT_ROOT / "models").glob("*.joblib"))
    assert models, "No saved models found in models/."


def test_figures_exist():
    figures = list((PROJECT_ROOT / "outputs" / "figures").glob("*.png"))
    assert figures, "No figures found in outputs/figures/."


if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASSED {test.__name__}")
    print(f"\n{len(tests)} output tests passed.")
