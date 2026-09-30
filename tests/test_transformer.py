"""Tests for the visit-day Transformer (research/transformer_model.py).

Run with `pytest tests/`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.gru_model import encode_patients, to_tensors
from research.transformer_model import VisitTransformer, days_before_cutoff
import torch
from tests.test_gru import TOKEN_TO_ID, make_events


def test_days_before_cutoff_counts_back_from_the_last_visit():
    encoded = encode_patients(make_events(), ["p1", "p2", "p3"], TOKEN_TO_ID)
    days = days_before_cutoff(encoded["gap_days"], encoded["lengths"], encoded["days_to_cutoff"])
    # p1: visits 60 days apart, the later one 661 days before the cutoff.
    assert days[0].tolist() == [721, 661]
    # p2: one visit 30 days before the cutoff; padding stays 0. p3 has no visits.
    assert days[1].tolist() == [30, 0]
    assert days[2].tolist() == [0, 0]


def test_transformer_ignores_padding_and_handles_empty_sequences():
    torch.manual_seed(0)
    encoded = encode_patients(make_events(), ["p1", "p2", "p3"], TOKEN_TO_ID)
    static = np.array([[0.5, 1.0], [-0.2, 0.0], [1.0, 1.0]], dtype=np.float32)
    model = VisitTransformer(vocab_size=encoded["counts"].shape[2], n_static=2).eval()
    padded = {**encoded,
              "counts": np.pad(encoded["counts"], ((0, 0), (0, 3), (0, 0))),
              "gap_days": np.pad(encoded["gap_days"], ((0, 0), (0, 3)))}
    with torch.no_grad():
        logits = model(to_tensors(encoded, static))
        logits_padded = model(to_tensors(padded, static))
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, logits_padded)


def test_transformer_module_never_loads_xgboost():
    script = "import sys, research.transformer_model; sys.exit('xgboost' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, timeout=120)
    assert result.returncode == 0, "research.transformer_model loaded xgboost"
