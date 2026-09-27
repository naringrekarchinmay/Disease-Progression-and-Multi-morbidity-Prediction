"""Time-decay GRU for multimorbidity progression (ISEM 735, Task 7).

Each patient's pre-cutoff events (src/sequences.py) become one step per visit day:
a vector of token counts for that day plus the days since the previous visit day.
The hidden state decays with elapsed time (as in GRU-D), then once more over the
gap from the last visit to the cutoff, before a small head adds age and sex.

    python -m research.gru_model --sample 5000   # debug run
    python -m research.gru_model                 # full at-risk cohort

Training uses the fit split, early stopping uses the validation split, and the
test split (the baseline's own) is scored once at the end.

This module must never load XGBoost. PyTorch and XGBoost each bundle their own
libomp, and on macOS the two in one process segfault or hang. The calibrated
baseline's test predictions are read from the file research.calibration writes.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_pipeline as rp
from research.calibration import (
    baseline_predictions_path,
    calibration_splits,
    plot_before_after,
    score_metrics,
    write_results,
)
from sklearn.isotonic import IsotonicRegression
from src.sequences import UNK_ID, build_vocabulary

CUTOFF = pd.Timestamp("2021-12-31")
WINDOW_START = pd.Timestamp("2018-01-01")
TARGET = "multimorbidity_progression"
SEQUENCES_PATH = rp.PROJECT_ROOT / "data" / "processed" / "sequences.parquet"
MODEL_PATH = rp.MODELS_DIR / f"{TARGET}_gru.pt"


def encode_patients(
    events: pd.DataFrame,
    patient_ids: list[str],
    token_to_id: dict[str, int],
    cutoff: pd.Timestamp = CUTOFF,
    window_start: pd.Timestamp = WINDOW_START,
) -> dict[str, np.ndarray]:
    """Turn event rows into padded per-visit-day arrays, in `patient_ids` order.

    Returns `counts` [patients, days, vocab] (token counts per visit day),
    `gap_days` [patients, days] (days since the previous visit day, 0 for the first),
    `lengths` [patients] and `days_to_cutoff` [patients] (from the last visit day,
    or from the start of the data window when a patient has no events).
    """
    position = pd.Series(np.arange(len(patient_ids)), index=pd.Index(patient_ids, name="patient_id"))
    ev = events[events["patient_id"].isin(position.index)].copy()
    ev["row"] = ev["patient_id"].map(position)
    ev["token_id"] = ev["token"].map(token_to_id).fillna(UNK_ID).astype(int)
    ev["day"] = ev.groupby("row")["date"].rank(method="dense").astype(int) - 1

    days = ev[["row", "day", "date"]].drop_duplicates().sort_values(["row", "day"])
    lengths = np.zeros(len(patient_ids), dtype=np.int64)
    lengths[days["row"].to_numpy()] = days["day"].to_numpy() + 1
    max_days = max(int(lengths.max()), 1)
    vocab_size = max(max(token_to_id.values(), default=UNK_ID), UNK_ID) + 1

    counts = np.zeros((len(patient_ids), max_days, vocab_size), dtype=np.uint8)
    np.add.at(counts, (ev["row"].to_numpy(), ev["day"].to_numpy(), ev["token_id"].to_numpy()), 1)

    gap_days = np.zeros((len(patient_ids), max_days), dtype=np.float32)
    gaps = days.groupby("row")["date"].diff().dt.days.fillna(0).to_numpy()
    gap_days[days["row"].to_numpy(), days["day"].to_numpy()] = gaps

    days_to_cutoff = np.full(len(patient_ids), (cutoff - window_start).days, dtype=np.float32)
    last = days.groupby("row")["date"].max()
    days_to_cutoff[last.index.to_numpy()] = (cutoff - last).dt.days.to_numpy()

    return {"counts": counts, "gap_days": gap_days, "lengths": lengths, "days_to_cutoff": days_to_cutoff}


def sequence_splits(features: pd.DataFrame, target: str = TARGET) -> dict[str, tuple[list[str], np.ndarray]]:
    """Patient ids and labels for fit, validation and test, identical to the calibrated baseline's."""
    return {
        name: (features.loc[X.index, "patient_id"].tolist(), y.to_numpy())
        for name, (X, y) in calibration_splits(features, target).items()
    }


def training_vocabulary(events: pd.DataFrame, fit_ids: list[str]) -> dict[str, int]:
    """Token ids built from fit-split patients only; anything else maps to UNK later."""
    vocab = build_vocabulary(events[events["patient_id"].isin(set(fit_ids))])
    return dict(zip(vocab["token"], vocab["token_id"]))


def to_tensors(encoded: dict[str, np.ndarray], static: np.ndarray) -> dict[str, torch.Tensor]:
    """Model inputs as tensors; the same keys work for a full split or a batch slice."""
    return {
        "counts": torch.as_tensor(encoded["counts"]),
        "gap_days": torch.as_tensor(encoded["gap_days"]),
        "lengths": torch.as_tensor(encoded["lengths"]),
        "days_to_cutoff": torch.as_tensor(encoded["days_to_cutoff"]),
        "static": torch.as_tensor(static),
    }


class TimeDecayGRU(nn.Module):
    """GRU over visit days whose hidden state decays with the time between visits.

    Before each visit day, and once more from the last visit to the cutoff, the
    hidden state is multiplied by exp(-relu(w * years + b)) per unit (GRU-D decay).
    Dropout sits before and inside the output head so Monte Carlo dropout can be
    used for uncertainty later.
    """

    def __init__(self, vocab_size: int, n_static: int, embed_dim: int = 32, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Linear(vocab_size, embed_dim)
        self.cell = nn.GRUCell(embed_dim + 1, hidden_dim)
        self.decay = nn.Linear(1, hidden_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim + n_static, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.hidden_dim = hidden_dim

    def _decay(self, h: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        years = (days / 365.25).unsqueeze(-1)
        return h * torch.exp(-torch.relu(self.decay(years)))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        counts = torch.log1p(batch["counts"].float())
        gaps = batch["gap_days"].float()
        lengths = batch["lengths"]
        h = counts.new_zeros(counts.shape[0], self.hidden_dim)
        for t in range(counts.shape[1]):
            active = (t < lengths).unsqueeze(-1)
            x = torch.cat([self.embed(counts[:, t]), torch.log1p(gaps[:, t] / 365.25).unsqueeze(-1)], dim=-1)
            h_new = self.cell(x, self._decay(h, gaps[:, t]))
            h = torch.where(active, h_new, h)
        h = self._decay(h, batch["days_to_cutoff"].float())
        return self.head(torch.cat([h, batch["static"].float()], dim=-1)).squeeze(-1)


def static_features(features: pd.DataFrame, ids: list[str], age_mean: float, age_std: float) -> np.ndarray:
    """Standardised age and a male indicator, in `ids` order."""
    rows = features.set_index("patient_id").loc[ids]
    return np.column_stack([(rows["age"] - age_mean) / age_std, (rows["sex"] == "M").astype(float)]).astype(np.float32)


def predict_proba(model: TimeDecayGRU, tensors: dict[str, torch.Tensor], batch_size: int = 1024) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = [
            torch.sigmoid(model({k: v[i : i + batch_size] for k, v in tensors.items()}))
            for i in range(0, len(tensors["lengths"]), batch_size)
        ]
    return torch.cat(out).numpy()


def train(
    model: TimeDecayGRU,
    train_tensors: dict[str, torch.Tensor],
    y_train: np.ndarray,
    val_tensors: dict[str, torch.Tensor],
    y_val: np.ndarray,
    seed: int,
    max_epochs: int = 40,
    patience: int = 4,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> tuple[TimeDecayGRU, pd.DataFrame]:
    """Adam with early stopping on validation log loss; returns the best-epoch model."""
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32)
    best_loss, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(y_train_t), generator=generator)
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model({k: v[idx] for k, v in train_tensors.items()}), y_train_t[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        p_val = torch.as_tensor(predict_proba(model, val_tensors)).clamp(1e-6, 1 - 1e-6)
        val_loss = nn.functional.binary_cross_entropy(p_val, y_val_t).item()
        history.append({"epoch": epoch, "train_loss": total / len(order), "val_loss": val_loss})
        print(f"  epoch {epoch:2d}  train {total / len(order):.4f}  val {val_loss:.4f}")
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the time-decay GRU on the at-risk cohort.")
    parser.add_argument("--sample", type=int, default=None, help="Debug on a random sample of N patients.")
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
    tensors = {}
    for name, (ids, _) in splits.items():
        encoded = encode_patients(events, ids, token_to_id)
        tensors[name] = to_tensors(encoded, static_features(features, ids, age_mean, age_std))
    vocab_size = max(t["counts"].shape[2] for t in tensors.values())
    for t in tensors.values():  # every split shares the fit vocabulary width
        t["counts"] = nn.functional.pad(t["counts"], (0, vocab_size - t["counts"].shape[2]))
    sizes = {name: len(ids) for name, (ids, _) in splits.items()}
    print(f"Splits {sizes} | vocabulary {len(token_to_id)} tokens | prep {time.perf_counter() - start:.1f}s")

    model = TimeDecayGRU(vocab_size=vocab_size, n_static=2)
    model, history = train(model, tensors["fit"], y_fit, tensors["validation"], splits["validation"][1], seed)
    trained = time.perf_counter()

    y_val, y_test = splits["validation"][1], splits["test"][1]
    p_val, p_test = predict_proba(model, tensors["validation"]), predict_proba(model, tensors["test"])
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val, y_val)
    scores = {"GRU": p_test, "GRU + isotonic": isotonic.predict(p_test)}

    if args.sample is None:
        baseline = pd.read_parquet(baseline_predictions_path(TARGET)).set_index("patient_id")
        test_ids = splits["test"][0]
        assert (baseline.loc[test_ids, "y_true"].to_numpy() == y_test).all(), "baseline test split differs"
        scores = {"XGBoost + isotonic": baseline.loc[test_ids, "y_score"].to_numpy(), **scores}

    table = pd.DataFrame([{"model": name, **score_metrics(y_test, p)} for name, p in scores.items()])
    print(table.round(4).to_string(index=False))
    print(f"Epochs {len(history)} (best {int(history.val_loss.idxmin()) + 1}) | train {trained - start:.0f}s total")

    if args.sample is not None:
        return
    torch.save(
        {"state_dict": model.state_dict(), "token_to_id": token_to_id, "vocab_size": vocab_size,
         "age_mean": age_mean, "age_std": age_std, "seed": seed},
        MODEL_PATH,
    )
    history.to_csv(rp.TABLES_DIR / "gru_training_history.csv", index=False)
    rp.save_metrics(table, rp.TABLES_DIR / "gru_vs_baseline.csv")
    plot_before_after(pd.Series(y_test), scores, TARGET, rp.FIGURES_DIR / "gru_vs_baseline_calibration.png",
                      title=f"GRU vs calibrated XGBoost: {TARGET}")

    log = table[table["model"].str.startswith("GRU")].assign(target=TARGET)
    write_results(log, task="7", seed=seed,
                  notes=f"time-decay GRU, fit split, early stop on validation, {len(history)} epochs")

if __name__ == "__main__":
    main()
