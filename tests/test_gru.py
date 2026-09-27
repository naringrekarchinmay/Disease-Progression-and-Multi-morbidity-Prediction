"""Tests for the GRU sequence model inputs, splits, and forward pass (research/gru_model.py).

Run with `pytest tests/`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_pipeline as rp
from research.calibration import calibration_splits
from research.gru_model import TimeDecayGRU, encode_patients, sequence_splits, to_tensors, training_vocabulary
import torch
from tests.test_calibration import make_fake_features
from src.sequences import UNK_ID


def make_events() -> pd.DataFrame:
    """p1 has two visit days, p2 one day with a token unseen in training, p3 none."""
    rows = [
        ("p1", "2020-01-10", "DX:I10"),
        ("p1", "2020-01-10", "LAB:hba1c:high"),
        ("p1", "2020-01-10", "LAB:hba1c:high"),
        ("p1", "2020-03-10", "MED:hypertension"),
        ("p2", "2021-12-01", "LAB:BNP:high"),
    ]
    events = pd.DataFrame(rows, columns=["patient_id", "date", "token"])
    events["date"] = pd.to_datetime(events["date"])
    return events


TOKEN_TO_ID = {"DX:I10": 2, "LAB:hba1c:high": 3, "MED:hypertension": 4}


def test_encode_patients_groups_tokens_by_visit_day():
    encoded = encode_patients(make_events(), ["p1", "p2", "p3"], TOKEN_TO_ID)
    counts, gaps = encoded["counts"], encoded["gap_days"]
    assert counts.shape[0] == 3 and counts.shape[1] == 2  # three patients, at most two visit days
    assert encoded["lengths"].tolist() == [2, 1, 0]

    # p1 day 1: one I10, two hba1c-high; day 2: one medication start, 60 days later.
    assert counts[0, 0, 2] == 1 and counts[0, 0, 3] == 2
    assert counts[0, 1, 4] == 1 and gaps[0].tolist() == [0, 60]
    # p2's token is not in the vocabulary, so it counts as UNK.
    assert counts[1, 0, UNK_ID] == 1
    # p3 has no events: an all-zero row.
    assert counts[2].sum() == 0

    # Days from the last visit to the 2021-12-31 cutoff; no visit counts as the whole window.
    assert encoded["days_to_cutoff"][:2].tolist() == [661, 30]
    assert encoded["days_to_cutoff"][2] == (pd.Timestamp("2021-12-31") - pd.Timestamp("2018-01-01")).days


TARGET = "multimorbidity_progression"


def test_splits_match_calibrated_baseline_and_vocab_uses_fit_only():
    features = make_fake_features()
    splits = sequence_splits(features)
    baseline = calibration_splits(features, TARGET)
    for name in ["fit", "validation", "test"]:
        ids, y = splits[name]
        X_base, y_base = baseline[name]
        assert ids == features.loc[X_base.index, "patient_id"].tolist()
        np.testing.assert_array_equal(y, y_base.to_numpy())

    test_only = splits["test"][0][0]
    events = pd.DataFrame(
        {
            "patient_id": [splits["fit"][0][0], test_only],
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "token": ["DX:I10", "DX:ONLY_IN_TEST"],
        }
    )
    vocab = training_vocabulary(events, splits["fit"][0])
    assert "DX:I10" in vocab and "DX:ONLY_IN_TEST" not in vocab


def test_model_output_ignores_padding_and_handles_empty_sequences():
    torch.manual_seed(0)
    encoded = encode_patients(make_events(), ["p1", "p2", "p3"], TOKEN_TO_ID)
    static = np.array([[0.5, 1.0], [-0.2, 0.0], [1.0, 1.0]], dtype=np.float32)
    model = TimeDecayGRU(vocab_size=encoded["counts"].shape[2], n_static=2).eval()

    padded = {**encoded,
              "counts": np.pad(encoded["counts"], ((0, 0), (0, 3), (0, 0))),
              "gap_days": np.pad(encoded["gap_days"], ((0, 0), (0, 3)))}
    with torch.no_grad():
        logits = model(to_tensors(encoded, static))
        logits_padded = model(to_tensors(padded, static))
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, logits_padded)


def test_gru_module_never_loads_xgboost():
    """PyTorch and XGBoost each bundle libomp; together in one process they crash on macOS.

    Runs in a fresh interpreter so modules loaded by other tests do not count.
    """
    script = "import sys, research.gru_model; sys.exit('xgboost' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, timeout=120)
    assert result.returncode == 0, "research.gru_model loaded xgboost"
