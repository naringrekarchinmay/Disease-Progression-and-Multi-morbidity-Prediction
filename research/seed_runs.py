"""Retrain the GRU and the Transformer over several seeds (ISEM 735, task 24).

Table 1 reported one seed per model. Here each sequence model is retrained with seeds
42, 1, 2, 3 and 4 on the same fit / validation / test split, so only initialisation, dropout
and batch order change. Test-split predictions go to data/processed/seed_predictions/ and are
scored, with the XGBoost seeds, by research/model_ci.py (a separate process, since torch and
XGBoost cannot share one).

    python -m research.seed_runs
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch

import run_pipeline as rp
from research.data import load_features
from research.gru_model import (SEQUENCES_PATH, TARGET, TimeDecayGRU, predict_proba, sequence_splits, split_tensors,
                                train, training_vocabulary)
from research.seed_paths import OUT_DIR, SEEDS, predictions_path
from research.transformer_model import VisitTransformer

MODELS = {"GRU": (TimeDecayGRU, 1e-3), "Transformer": (VisitTransformer, 1e-3)}  # lr as in tasks 7 and 13


def main() -> None:
    torch.set_num_threads(os.cpu_count() or 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    features = load_features()
    events = pd.read_parquet(SEQUENCES_PATH, columns=["patient_id", "date", "token"])
    splits = sequence_splits(features)
    fit_ids, y_fit = splits["fit"]
    token_to_id = training_vocabulary(events, fit_ids)
    fit_rows = features.set_index("patient_id").loc[fit_ids]
    age_mean, age_std = float(fit_rows["age"].mean()), float(fit_rows["age"].std())
    vocab_size = max(token_to_id.values()) + 1
    tensors = {name: split_tensors(features, events, ids, token_to_id, vocab_size, age_mean, age_std)
               for name, (ids, _) in splits.items()}
    test_ids, y_test = splits["test"]
    print(f"prep {time.perf_counter() - start:.0f}s", flush=True)

    for name, (cls, lr) in MODELS.items():
        for seed in SEEDS:
            path = predictions_path(name, seed)
            if path.exists():
                print(f"{name} seed {seed}: exists, skipped", flush=True)
                continue
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = cls(vocab_size=vocab_size, n_static=2)
            model, history = train(model, tensors["fit"], y_fit, tensors["validation"], splits["validation"][1],
                                   seed, lr=lr)
            p_test = predict_proba(model, tensors["test"])
            pd.DataFrame({"patient_id": test_ids, "y_true": y_test, "y_score": p_test}).to_parquet(path, index=False)
            print(f"{name} seed {seed}: {len(history)} epochs | {time.perf_counter() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
