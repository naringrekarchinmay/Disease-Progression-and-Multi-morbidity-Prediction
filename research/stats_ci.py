"""Patient-level bootstrap and DeLong tests for paired model comparisons (ISEM 735, task 24).

Part 3 feedback: Table 1 rests on AUROC differences of 0.007 to 0.014 and Table 2's
threshold-over-random gap is 0.002, all from one seed on one test set. Everything here
compares scores on the same patients, so the comparisons are paired: each bootstrap
resample draws patients once and scores every model on that same draw.

No XGBoost or torch here, so the tests can import it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import brier_score_loss, roc_auc_score

N_BOOT = 2000
ALPHA = 0.05


def delong(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    """Paired DeLong test of AUROC(a) - AUROC(b) (DeLong et al., 1988; fast form of Sun & Xu, 2014)."""
    y = np.asarray(y).astype(bool)
    scores = np.vstack([score_a, score_b]).astype(float)
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([rankdata(p) for p in pos])
    ty = np.array([rankdata(q) for q in neg])
    tz = np.array([rankdata(s) for s in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var, 0.0)))
    z = diff / se if se > 0 else 0.0
    half = norm.ppf(1 - ALPHA / 2) * se
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": float(diff), "se": se,
            "ci_low": float(diff - half), "ci_high": float(diff + half), "p_value": float(2 * norm.sf(abs(z)))}


def bootstrap_indices(n: int, n_boot: int = N_BOOT, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _metric(name: str):
    return {"roc_auc": roc_auc_score, "brier": brier_score_loss}[name]


def bootstrap_metrics(y: np.ndarray, scores: dict[str, np.ndarray], idx: np.ndarray,
                      metric: str = "roc_auc") -> pd.DataFrame:
    """One row per resample, one column per score set, all on the same patients."""
    fn, y = _metric(metric), np.asarray(y)
    return pd.DataFrame({name: [fn(y[i], s[i]) for i in idx] for name, s in
                         ((k, np.asarray(v)) for k, v in scores.items())})


def summarize(point: float, draws: np.ndarray) -> dict:
    lo, hi = np.percentile(draws, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"estimate": float(point), "ci_low": float(lo), "ci_high": float(hi), "boot_se": float(np.std(draws))}


def paired_difference(y: np.ndarray, a: np.ndarray | list[np.ndarray], b: np.ndarray | list[np.ndarray],
                      idx: np.ndarray, metric: str = "roc_auc") -> dict:
    """Bootstrap CI of metric(a) - metric(b) on shared resamples.

    `a` or `b` may be a list of score sets (for example five random-schedule seeds); the metric
    is then averaged over the list within each resample, matching how Table 2 averages them.
    """
    fn, y = _metric(metric), np.asarray(y)
    a_list = a if isinstance(a, list) else [a]
    b_list = b if isinstance(b, list) else [b]

    def value(rows) -> float:
        return float(np.mean([fn(y[rows], s[rows]) for s in a_list])
                     - np.mean([fn(y[rows], s[rows]) for s in b_list]))

    draws = np.array([value(i) for i in idx])
    out = summarize(value(np.arange(len(y))), draws)
    out["share_above_zero"] = float(np.mean(draws > 0))
    return out
