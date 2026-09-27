"""Tests for Monte Carlo dropout uncertainty (research/uncertainty.py).

Run with `pytest tests/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.gru_model import TimeDecayGRU, encode_patients, to_tensors
from research.uncertainty import lab_visits_before_cutoff, mc_dropout_predict
import torch
from tests.test_gru import TOKEN_TO_ID, make_events


def make_model_and_inputs():
    torch.manual_seed(0)
    encoded = encode_patients(make_events(), ["p1", "p2", "p3"], TOKEN_TO_ID)
    static = np.array([[0.5, 1.0], [-0.2, 0.0], [1.0, 1.0]], dtype=np.float32)
    model = TimeDecayGRU(vocab_size=encoded["counts"].shape[2], n_static=2).eval()
    return model, to_tensors(encoded, static)


def test_mc_dropout_passes_vary_are_reproducible_and_restore_eval_mode():
    model, tensors = make_model_and_inputs()
    draws = mc_dropout_predict(model, tensors, passes=30, seed=42)
    assert draws.shape == (30, 3)
    assert ((draws > 0) & (draws < 1)).all()
    assert draws.std(axis=0).min() > 0  # dropout is active, so passes differ
    np.testing.assert_array_equal(draws, mc_dropout_predict(model, tensors, passes=30, seed=42))
    assert not model.training


def test_lab_visits_count_distinct_lab_days_per_patient():
    events = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p1", "p2", "p2"],
            "date": pd.to_datetime(["2020-01-10", "2020-01-10", "2021-02-01", "2020-05-05", "2020-06-06"]),
            "event_type": ["lab", "lab", "lab", "diagnosis", "lab"],
        }
    )
    counts = lab_visits_before_cutoff(events, ["p2", "p3", "p1"])
    assert counts.tolist() == [1, 0, 2]
