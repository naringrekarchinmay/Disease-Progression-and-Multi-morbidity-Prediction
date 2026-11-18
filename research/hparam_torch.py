"""Hyperparameter search and five-seed refits for the GRU and the Transformer (ISEM 735, task 29).

Protocol: research/hparam_notes.md section 1. Each model gets 50 Optuna TPE trials scored by
validation log loss, trial 0 = its default, median pruning on per-epoch validation loss. The best
configuration is then retrained with seeds 42, 1, 2, 3, 4 and its test predictions saved for
research/hparam_xgb.py, which builds the tuned Table 1.

    python -m research.hparam_torch                 # both models
    python -m research.hparam_torch --model GRU

This module uses torch, so it must never load XGBoost.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import optuna
import pandas as pd
import torch

from research.data import load_features
from research.gru_model import (SEQUENCES_PATH, TimeDecayGRU, predict_proba, sequence_splits, split_tensors, train,
                                training_vocabulary)
from research.hparam_paths import (N_TRIALS, OUT_DIR, TRAIN_SEED, TRAINING_KEYS, load_best, make_study,
                                   predictions_path, save_study, suggest)
from research.seed_paths import SEEDS
from research.transformer_model import VisitTransformer

CLASSES = {"GRU": TimeDecayGRU, "Transformer": VisitTransformer}


def prepare(sample: int | None = None) -> dict:
    features = load_features(sample)
    events = pd.read_parquet(SEQUENCES_PATH, columns=["patient_id", "date", "token"])
    splits = sequence_splits(features)
    fit_ids, _ = splits["fit"]
    token_to_id = training_vocabulary(events, fit_ids)
    fit_rows = features.set_index("patient_id").loc[fit_ids]
    age_mean, age_std = float(fit_rows["age"].mean()), float(fit_rows["age"].std())
    vocab_size = max(token_to_id.values()) + 1
    tensors = {name: split_tensors(features, events, ids, token_to_id, vocab_size, age_mean, age_std)
               for name, (ids, _) in splits.items()}
    return {"splits": splits, "tensors": tensors, "vocab_size": vocab_size}


def fit(model_name: str, params: dict, data: dict, seed: int, epoch_callback=None):
    """Train one configuration; returns the best-epoch model and its training history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    arch = {k: v for k, v in params.items() if k not in TRAINING_KEYS}
    training = {k: params[k] for k in TRAINING_KEYS}
    model = CLASSES[model_name](vocab_size=data["vocab_size"], n_static=2, **arch)
    splits, tensors = data["splits"], data["tensors"]
    return train(model, tensors["fit"], splits["fit"][1], tensors["validation"], splits["validation"][1], seed,
                 epoch_callback=epoch_callback, verbose=False, **training)


def objective(model_name: str, data: dict):
    def run(trial: optuna.Trial) -> float:
        params = suggest(trial, model_name)

        def report(epoch: int, val_loss: float) -> None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _, history = fit(model_name, params, data, TRAIN_SEED, epoch_callback=report)
        trial.set_user_attr("epochs", len(history))
        return float(history["val_loss"].min())
    return run


def search(model_name: str, data: dict, n_trials: int = N_TRIALS) -> dict:
    study = make_study(model_name)
    start = time.perf_counter()

    def progress(study, trial):
        best = study.best_value if any(t.value is not None for t in study.trials) else float("nan")
        print(f"  {model_name} trial {trial.number:2d} {trial.state.name:8s} value {trial.value} "
              f"best {best:.4f} | {time.perf_counter() - start:.0f}s", flush=True)

    study.optimize(objective(model_name, data), n_trials=n_trials, callbacks=[progress])
    return save_study(study, model_name)


def refit(model_name: str, data: dict) -> None:
    params = load_best(model_name)
    test_ids, y_test = data["splits"]["test"]
    for seed in SEEDS:
        path = predictions_path(model_name, seed)
        model, history = fit(model_name, params, data, seed)
        p_test = predict_proba(model, data["tensors"]["test"])
        pd.DataFrame({"patient_id": test_ids, "y_true": y_test, "y_score": p_test}).to_parquet(path, index=False)
        print(f"  {model_name} refit seed {seed}: {len(history)} epochs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=list(CLASSES), action="append")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    data = prepare()
    for name in args.model or list(CLASSES):
        best = search(name, data, args.trials)
        print(f"{name}: best trial {best['trial']} val log loss {best['val_log_loss']:.4f} "
              f"(default {best['default_val_log_loss']:.4f}) {best['params']} | {time.perf_counter() - start:.0f}s",
              flush=True)
        refit(name, data)
    print(f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
