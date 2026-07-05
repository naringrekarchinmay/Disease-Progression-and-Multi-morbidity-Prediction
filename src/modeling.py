"""Model training helpers for baseline and advanced classifiers."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical model features."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def make_baseline_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Return simple baseline models that should be trained before advanced models."""
    preprocessor = build_preprocessor(X)
    return {
        "dummy_most_frequent": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X)),
                (
                    "model",
                    LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None),
                ),
            ]
        ),
    }


def make_advanced_models(
    X: pd.DataFrame, random_state: int = 42, pos_weight: float | None = None
) -> dict[str, Pipeline]:
    """Return stronger models, using optional libraries only if installed.

    ``pos_weight`` is the ratio of negative to positive examples. When given it
    is passed to XGBoost as ``scale_pos_weight`` so it handles class imbalance
    like the ``class_weight="balanced"`` option used by the other models.
    """
    models: dict[str, Pipeline] = {
        "random_forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        scale_pos_weight=pos_weight if pos_weight is not None else 1.0,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except Exception:
        # xgboost can fail beyond ImportError, e.g. a missing OpenMP runtime.
        pass

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X)),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except Exception:
        # lightgbm can also fail at import time when OpenMP is unavailable.
        pass

    return models


def fit_models(models: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Pipeline]:
    """Fit a dictionary of sklearn-compatible models."""
    fitted = {}
    for name, model in models.items():
        fitted[name] = model.fit(X_train, y_train)
    return fitted


def save_model(model: Pipeline, path: str | Path) -> Path:
    """Save a fitted model pipeline with joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def load_model(path: str | Path):
    """Load a fitted model pipeline with joblib."""
    return joblib.load(path)

