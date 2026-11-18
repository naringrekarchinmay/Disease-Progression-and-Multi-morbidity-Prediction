"""Tests for the Table 1 hyperparameter search (research/hparam_*.py, task 29).

Run with `pytest tests/`.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.gru_model import TimeDecayGRU, encode_patients, to_tensors, train
from research.hparam_paths import DEFAULTS, TRAINING_KEYS, make_study, suggest
from research.transformer_model import VisitTransformer
import torch
from tests.test_gru import TOKEN_TO_ID, make_events

CLASSES = {"GRU": TimeDecayGRU, "Transformer": VisitTransformer}


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)


@pytest.mark.parametrize("module, banned", [("research.hparam_paths", "torch"), ("research.hparam_paths", "xgboost"),
                                            ("research.hparam_torch", "xgboost"), ("research.hparam_xgb", "torch")])
def test_modules_keep_torch_and_xgboost_apart(module, banned):
    result = _run(f"import sys, {module}; sys.exit('{banned}' in sys.modules)")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("model", ["GRU", "Transformer"])
def test_defaults_match_the_settings_table_1_used(model):
    ctor = inspect.signature(CLASSES[model].__init__).parameters
    for key, value in DEFAULTS[model].items():
        if key not in TRAINING_KEYS:
            assert ctor[key].default == value, key
    fit = inspect.signature(train).parameters
    for key in TRAINING_KEYS:
        assert fit[key].default == DEFAULTS[model][key], key


@pytest.mark.parametrize("model", list(DEFAULTS))
def test_defaults_lie_in_the_search_space_and_are_trial_zero(model):
    assert suggest(optuna.trial.FixedTrial(DEFAULTS[model]), model) == DEFAULTS[model]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    assert suggest(make_study(model).ask(), model) == DEFAULTS[model]


@pytest.mark.parametrize("dim, heads, layers", [(d, h, l) for d in [32, 64, 128] for h in [2, 4] for l in [1, 2, 3]])
def test_every_transformer_shape_in_the_space_runs(dim, heads, layers):
    torch.manual_seed(0)
    encoded = encode_patients(make_events(), ["p1", "p2", "p3"], TOKEN_TO_ID)
    static = np.zeros((3, 2), dtype=np.float32)
    model = VisitTransformer(vocab_size=encoded["counts"].shape[2], n_static=2, dim=dim, heads=heads, layers=layers)
    with torch.no_grad():
        assert torch.isfinite(model.eval()(to_tensors(encoded, static))).all()


def test_epoch_callback_sees_every_epoch_and_can_stop_training():
    torch.manual_seed(0)
    encoded = encode_patients(make_events(), ["p1", "p2", "p3"], TOKEN_TO_ID)
    tensors = to_tensors(encoded, np.zeros((3, 2), dtype=np.float32))
    y = np.array([1, 0, 1])
    seen = []

    def stop_at_two(epoch, val_loss):
        seen.append(epoch)
        if epoch == 2:
            raise optuna.TrialPruned()

    model = TimeDecayGRU(vocab_size=encoded["counts"].shape[2], n_static=2)
    with pytest.raises(optuna.TrialPruned):
        train(model, tensors, y, tensors, y, seed=0, max_epochs=10, patience=10, verbose=False,
              epoch_callback=stop_at_two)
    assert seen == [1, 2]


def test_tuned_xgboost_is_unweighted():
    script = ("from tests.test_calibration import make_fake_features\n"
              "from research.hparam_xgb import make_xgboost\n"
              "from research.hparam_paths import DEFAULTS\n"
              "f = make_fake_features(); X = f[['age', 'signal', 'sex']]; y = f['multimorbidity_progression']\n"
              "m = make_xgboost(X, DEFAULTS['XGBoost'], 0).fit(X, y)\n"
              "assert m.named_steps['model'].get_params()['scale_pos_weight'] == 1.0\n"
              "assert 0 < m.predict_proba(X)[:, 1].mean() < 1\n")
    result = _run(script)
    assert result.returncode == 0, result.stderr
