"""Seeds and prediction paths shared by research/seed_runs.py (torch) and research/model_ci.py
(XGBoost). Kept free of both libraries so either side can import it (see tests/conftest.py)."""

from __future__ import annotations

import run_pipeline as rp

SEEDS = [42, 1, 2, 3, 4]
OUT_DIR = rp.PROJECT_ROOT / "data" / "processed" / "seed_predictions"


def predictions_path(model: str, seed: int):
    return OUT_DIR / f"{model.lower()}_seed{seed}.parquet"
