"""Visit-day Transformer for multimorbidity progression (ISEM 735, proposal weeks 4-5).

The counterpart to research/gru_model.py on identical inputs, split, seed and vocabulary.
Each visit day is one token (its token counts, embedded); instead of a decaying hidden
state, each day carries a sinusoidal encoding of how long before the cutoff it happened,
as Delphi encodes age. A summary token placed at the cutoff attends over the visit days
and feeds the same age-and-sex head as the GRU.

    python -m research.transformer_model --sample 5000   # debug run
    python -m research.transformer_model                 # full at-risk cohort

Like research.gru_model, this module must never load XGBoost (see that module).
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_pipeline as rp


def days_before_cutoff(gap_days, lengths, days_to_cutoff) -> torch.Tensor:
    """Days from each visit day to the cutoff, [patients, days]; 0 on padding.

    The last visit is `days_to_cutoff` before the cutoff and each earlier visit adds the
    gap that followed it.
    """
    gaps = torch.as_tensor(gap_days, dtype=torch.float32)
    lengths = torch.as_tensor(lengths)
    valid = torch.arange(gaps.shape[1]) < lengths.unsqueeze(-1)
    gaps = gaps * valid
    cumulative = gaps.cumsum(dim=1)
    total = cumulative[:, -1:]
    days = torch.as_tensor(days_to_cutoff, dtype=torch.float32).unsqueeze(-1) + total - cumulative
    return days * valid


class TimeEncoding(nn.Module):
    """Sinusoids of years before the cutoff, periods from about a month to 20 years."""

    def __init__(self, dim: int):
        super().__init__()
        periods = torch.logspace(math.log10(0.1), math.log10(20.0), dim // 2)
        self.register_buffer("frequencies", 2 * math.pi / periods)
        self.proj = nn.Linear(2 * (dim // 2), dim)

    def forward(self, days: torch.Tensor) -> torch.Tensor:
        angles = (days / 365.25).unsqueeze(-1) * self.frequencies
        return self.proj(torch.cat([angles.sin(), angles.cos()], dim=-1))


class VisitTransformer(nn.Module):
    """Transformer encoder over visit days with a summary token placed at the cutoff.

    Visit days are embedded from their token counts plus a time encoding; padded days are
    masked out of attention, so a patient's output does not depend on padding. The
    summary token's output and the static features (age, sex) feed the same head as the
    GRU, dropout included, so Monte Carlo dropout works the same way.
    """

    def __init__(self, vocab_size: int, n_static: int, dim: int = 64, heads: int = 4, layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Linear(vocab_size, dim)
        self.time = TimeEncoding(dim)
        self.summary = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim + n_static, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        counts = torch.log1p(batch["counts"].float())
        lengths = batch["lengths"]
        days = days_before_cutoff(batch["gap_days"], lengths, batch["days_to_cutoff"])
        visits = self.embed(counts) + self.time(days)
        summary = self.summary.expand(len(counts), 1, -1) + self.time(torch.zeros(len(counts), 1))
        tokens = torch.cat([summary, visits], dim=1)
        padding = torch.arange(counts.shape[1]) >= lengths.unsqueeze(-1)
        mask = torch.cat([torch.zeros(len(counts), 1, dtype=torch.bool), padding], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=mask)
        return self.head(torch.cat([encoded[:, 0], batch["static"].float()], dim=-1)).squeeze(-1)


MODEL_PATH = rp.MODELS_DIR / "multimorbidity_progression_transformer.pt"
PREDICTIONS_PATH = rp.PROJECT_ROOT / "data" / "processed" / "transformer_test_predictions.parquet"


def main() -> None:
    from sklearn.isotonic import IsotonicRegression

    from research.calibration import baseline_predictions_path, plot_before_after, score_metrics, write_results
    from research.equity import performance_by_group, subgroups
    from research.gru_model import (
        SEQUENCES_PATH, TARGET, predict_proba, sequence_splits, split_tensors, train, training_vocabulary,
    )
    from research.uncertainty import PREDICTIONS_PATH as GRU_PREDICTIONS_PATH

    parser = argparse.ArgumentParser(description="Train the visit-day Transformer on the at-risk cohort.")
    parser.add_argument("--sample", type=int, default=None, help="Debug on a random sample of N patients.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (chosen on validation loss).")
    args = parser.parse_args()
    seed = rp.RANDOM_STATE
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(os.cpu_count() or 1)
    start = time.perf_counter()

    features = rp.load_feature_table(args.sample)
    events = pd.read_parquet(SEQUENCES_PATH, columns=["patient_id", "date", "token"])
    splits = sequence_splits(features)
    fit_ids, y_fit = splits["fit"]
    token_to_id = training_vocabulary(events, fit_ids)
    fit_rows = features.set_index("patient_id").loc[fit_ids]
    age_mean, age_std = float(fit_rows["age"].mean()), float(fit_rows["age"].std())
    vocab_size = max(token_to_id.values()) + 1
    tensors = {name: split_tensors(features, events, ids, token_to_id, vocab_size, age_mean, age_std)
               for name, (ids, _) in splits.items()}
    n_params = sum(p.numel() for p in VisitTransformer(vocab_size, 2).parameters())
    print(f"Splits { {k: len(v[0]) for k, v in splits.items()} } | vocabulary {len(token_to_id)} | params {n_params:,}")

    model = VisitTransformer(vocab_size=vocab_size, n_static=2)
    model, history = train(model, tensors["fit"], y_fit, tensors["validation"], splits["validation"][1], seed, lr=args.lr)
    trained = time.perf_counter()

    test_ids, y_test = splits["test"]
    p_val, p_test = predict_proba(model, tensors["validation"]), predict_proba(model, tensors["test"])
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val, splits["validation"][1])
    scores = {"Transformer": p_test, "Transformer + isotonic": isotonic.predict(p_test)}
    if args.sample is None:
        gru = pd.read_parquet(GRU_PREDICTIONS_PATH).set_index("patient_id").loc[test_ids]
        xgb = pd.read_parquet(baseline_predictions_path(TARGET)).set_index("patient_id").loc[test_ids]
        assert (gru["y_true"].to_numpy() == y_test).all() and (xgb["y_true"].to_numpy() == y_test).all()
        scores = {"XGBoost + isotonic": xgb["y_score"].to_numpy(), "GRU": gru["deterministic"].to_numpy(), **scores}

    table = pd.DataFrame([{"model": name, **score_metrics(y_test, p)} for name, p in scores.items()])
    print(table.round(4).to_string(index=False))
    print(f"lr {args.lr} | epochs {len(history)} (best {int(history.val_loss.idxmin()) + 1}, "
          f"val loss {history.val_loss.min():.4f}) | train {trained - start:.0f}s")
    if args.sample is not None:
        return

    patients = pd.read_csv(rp.PROJECT_ROOT / "data" / "patients.csv", usecols=["patient_id", "age", "sex", "insurance_type"])
    by_age = pd.DataFrame({"patient_id": test_ids, "y_true": y_test}).merge(subgroups(patients), on="patient_id")
    age_rows = []
    for name in ["XGBoost + isotonic", "GRU", "Transformer"]:
        by_age["score"] = scores[name]
        age_rows.append(performance_by_group(by_age, "age_band", "y_true", "score", n_boot=200, seed=seed).assign(model=name))
    age_table = pd.concat(age_rows, ignore_index=True)
    print(age_table.pivot(index="group", columns="model", values="auroc").round(3).to_string())

    torch.save({"state_dict": model.state_dict(), "token_to_id": token_to_id, "vocab_size": vocab_size,
                "age_mean": age_mean, "age_std": age_std, "seed": seed}, MODEL_PATH)
    pd.DataFrame({"patient_id": test_ids, "y_true": y_test, "y_score": p_test}).to_parquet(PREDICTIONS_PATH, index=False)
    history.to_csv(rp.TABLES_DIR / "transformer_training_history.csv", index=False)
    rp.save_metrics(table, rp.TABLES_DIR / "transformer_vs_baselines.csv")
    rp.save_metrics(age_table, rp.TABLES_DIR / "transformer_auroc_by_age.csv")
    plot_before_after(pd.Series(y_test), {k: scores[k] for k in ["XGBoost + isotonic", "GRU", "Transformer"]}, TARGET,
                      rp.FIGURES_DIR / "transformer_vs_baselines_calibration.png",
                      title="Multimorbidity progression: three models, test set")
    write_results(table[table["model"].str.startswith("Transformer")].assign(target=TARGET), task="13", seed=seed,
                  notes=f"visit-day Transformer, {n_params:,} params, lr {args.lr} chosen on validation loss, "
                        f"fit split, early stop on validation, {len(history)} epochs")


if __name__ == "__main__":
    main()
