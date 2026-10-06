"""Measurement scheduling under masking (ISEM 735, proposal weeks 7-8).

Decision points are each test patient's observed pre-cutoff lab visits, in date order.
A schedule keeps or skips each one; masking can only drop measurements, so it scores
schedules sparser than observed care. After a schedule runs, each patient's features
are rebuilt at the cutoff from the kept labs and scored with the calibrated XGBoost,
the predictor the proposal's week 5 rule selects (both sequence models trail it).

Controllers: an uncertainty-threshold rule (keep a visit when the risk known before it
is within tau of 50%), plus the proposal's controls: keep all (observed care), keep
none, random visits matched on budget, and fixed intervals.

    python -m research.scheduling

This module loads XGBoost, so it must never import torch (see research/gru_model.py).
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_pipeline as rp
from src.features import TARGET_COLUMNS, build_patient_features

CUTOFF = pd.Timestamp("2021-12-31")
INTERVAL_TOLERANCE_DAYS = 30
TARGET = "multimorbidity_progression"
SCORE_DATE = pd.Timestamp("2022-01-01")  # features "as of" the day after the cutoff
INPUT_YEARS = 4  # 2018-01-01 to 2021-12-31: the window the decision points fall in
TAUS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
RANDOM_SEEDS = [1, 2, 3, 4, 5]


def lab_opportunities(labs: pd.DataFrame, patient_ids: list[str]) -> pd.DataFrame:
    """Distinct pre-cutoff lab-visit days per patient: patient_id, date, k (0-based order)."""
    days = labs.loc[labs["patient_id"].isin(set(patient_ids)) & (labs["test_date"] <= CUTOFF),
                    ["patient_id", "test_date"]].drop_duplicates()
    days = days.rename(columns={"test_date": "date"}).sort_values(["patient_id", "date"]).reset_index(drop=True)
    days["k"] = days.groupby("patient_id").cumcount()
    return days


def fixed_interval_keep(opps: pd.DataFrame, interval_days: int,
                        tolerance_days: int = INTERVAL_TOLERANCE_DAYS) -> pd.Series:
    """Keep the first visit, then any visit at least `interval_days - tolerance` after the last kept one."""
    keep = np.zeros(len(opps), dtype=bool)
    last_patient, last_kept = None, None
    for i, (patient, date) in enumerate(zip(opps["patient_id"], opps["date"])):
        if patient != last_patient:
            last_patient, last_kept = patient, None
        if last_kept is None or (date - last_kept).days >= interval_days - tolerance_days:
            keep[i], last_kept = True, date
    return pd.Series(keep, index=opps.index)


def random_keep(opps: pd.DataFrame, n_keep: int, seed: int = 42) -> pd.Series:
    """Keep exactly `n_keep` visits chosen uniformly at random (the budget-matched control)."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(opps), dtype=bool)
    keep[rng.choice(len(opps), size=n_keep, replace=False)] = True
    return pd.Series(keep, index=opps.index)


def threshold_policy(opps: pd.DataFrame, risk_fn, tau: float) -> pd.Series:
    """Keep visit k when the risk known just before it is within `tau` of 50%.

    Decisions run in visit order: `risk_fn(opps, keep, k)` sees only the choices already
    made for earlier visits and returns one risk per patient that has a visit k.
    """
    keep = pd.Series(False, index=opps.index)
    for k in range(int(opps["k"].max()) + 1):
        at_k = opps["k"] == k
        risk = risk_fn(opps, keep, k)
        close = (risk - 0.5).abs() <= tau
        keep[at_k] = opps.loc[at_k, "patient_id"].map(close).fillna(False).to_numpy(dtype=bool)
    return keep


def features_as_of(data: dict[str, pd.DataFrame], as_of: pd.Series, labs: pd.DataFrame,
                   columns: list[str]) -> pd.DataFrame:
    """Feature rows using only events strictly before each patient's `as_of` date.

    `as_of` maps patient_id to a date; `labs` is the lab table the schedule kept.
    Columns follow `columns` (the training feature table's), so condition and
    medication flags absent from this subset are 0 and absent lab values are missing.
    """
    ids = set(as_of.index)

    def before(table: pd.DataFrame, date_col: str) -> pd.DataFrame:
        part = table[table["patient_id"].isin(ids)]
        return part[part[date_col] < part["patient_id"].map(as_of)]

    features = build_patient_features(
        patients=data["patients"][data["patients"]["patient_id"].isin(ids)],
        diagnoses=before(data["diagnoses"], "visit_date"),
        lab_results=before(labs, "test_date"),
        medications=before(data["medications"], "start_date"),
        outcomes=before(data["outcomes"], "admission_date"),
        cutoff_date=str(CUTOFF.date()),
    )
    features = features.set_index("patient_id").reindex(list(as_of.index))
    missing = [c for c in columns[1:] if c not in features.columns]
    zero_fill = [c for c in missing if c.startswith(("baseline_dx_", "med_indication_"))]
    features = features.reindex(columns=columns[1:])
    features[zero_fill] = 0
    return features.reset_index()


class Scorer:
    """Scores a schedule on the test split with the calibrated XGBoost."""

    def __init__(self):
        import joblib

        from research.calibration import calibration_splits, score_metrics
        from research.equity import subgroups
        from src.data_cleaning import clean_all, load_raw_data

        self.score_metrics = score_metrics
        features = rp.load_feature_table()
        X_test, y_test = calibration_splits(features, TARGET)["test"]
        self.test_ids = features.loc[X_test.index, "patient_id"].tolist()
        self.y = y_test.to_numpy()
        self.columns = [c for c in features.columns if c not in TARGET_COLUMNS]
        self.model = joblib.load(rp.MODELS_DIR / f"{TARGET}_xgboost_isotonic.joblib")
        raw = load_raw_data()
        ids = set(self.test_ids)
        self.data = clean_all({name: t[t["patient_id"].isin(ids)] for name, t in raw.items()})
        self.labs = self.data["lab_results"]
        self.opps = lab_opportunities(self.labs, self.test_ids)
        self.age_band = subgroups(raw["patients"]).set_index("patient_id")["age_band"]
        summary = pd.read_csv(rp.TABLES_DIR / "cost_model_summary.csv")
        self.measurement_cost = float(summary["measurement_cost"].iloc[0])

    def kept_labs(self, keep: pd.Series) -> pd.DataFrame:
        kept = self.opps.loc[keep.to_numpy(), ["patient_id", "date"]].rename(columns={"date": "test_date"})
        return self.labs.merge(kept, on=["patient_id", "test_date"])

    def predict(self, as_of: pd.Series, labs: pd.DataFrame) -> pd.Series:
        rows = features_as_of(self.data, as_of, labs, self.columns)
        X = rows.drop(columns="patient_id")
        return pd.Series(self.model.predict_proba(X)[:, 1], index=rows["patient_id"])

    def risk_fn(self, opps: pd.DataFrame, keep: pd.Series, k: int) -> pd.Series:
        at_k = opps[opps["k"] == k]
        return self.predict(pd.Series(at_k["date"].to_numpy(), index=at_k["patient_id"]), self.kept_labs(keep))

    def score(self, keep: pd.Series) -> dict:
        as_of = pd.Series(SCORE_DATE, index=self.test_ids)
        risk = self.predict(as_of, self.kept_labs(keep)).loc[self.test_ids].to_numpy()
        n_kept = int(keep.sum())
        per_year = n_kept / len(self.test_ids) / INPUT_YEARS
        kept_by_age = keep.groupby(self.opps["patient_id"].map(self.age_band), observed=True).mean()
        overall = keep.mean() if len(keep) else 0.0
        result = {
            "visits_kept": n_kept,
            "share_of_observed": n_kept / len(self.opps),
            "measurements_per_patient_year": per_year,
            "spend_per_patient_year": per_year * self.measurement_cost,
            **self.score_metrics(self.y, risk),
        }
        for band, rate in kept_by_age.items():
            result[f"keep_ratio_age_{band}"] = rate / overall if overall else np.nan
        return result


def plot_frontier(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    styles = {"threshold": ("#C44E52", "o-"), "random (matched)": ("#4C72B0", "s--"),
              "fixed interval": ("#55A868", "D"), "keep all (observed)": ("black", "*"), "keep none": ("grey", "X")}
    for ax, metric, label in [(axes[0], "roc_auc", "AUROC"), (axes[1], "brier", "Brier score (lower is better)")]:
        for controller, (color, style) in styles.items():
            part = table[table["controller"] == controller].sort_values("spend_per_patient_year")
            if part.empty:
                continue
            err = part.get(f"{metric}_sd")
            ax.errorbar(part["spend_per_patient_year"], part[metric], yerr=err if err is not None and err.notna().any() else None,
                        fmt=style, color=color, markersize=7, capsize=2, label=controller)
            if controller == "fixed interval":
                for _, r in part.iterrows():
                    ax.annotate(r["setting"], (r["spend_per_patient_year"], r[metric]), fontsize=8,
                                xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Measurement spend per patient-year, 2018-2021 ($)")
        ax.set_ylabel(label)
    # Fixed ranges: auto-scaling would zoom into noise and make the flat frontier look shaped.
    axes[0].set_ylim(0.84, 0.88)
    axes[1].set_ylim(0.135, 0.155)
    spread = table["roc_auc"].max() - table["roc_auc"].min()
    axes[0].set_title(f"Budget frontier under masking: all schedules within {spread:.4f} AUROC")
    axes[1].set_title("Calibration side of the frontier")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(rp.FIGURES_DIR / "scheduling_frontier_masking.png", dpi=150)
    plt.close(fig)


def main() -> None:
    from research.calibration import write_results

    start = time.perf_counter()
    scorer = Scorer()
    opps = scorer.opps
    print(f"Test patients {len(scorer.test_ids):,} | decision points {len(opps):,} | "
          f"max visits per patient {int(opps['k'].max()) + 1} | setup {time.perf_counter() - start:.0f}s")

    rows = []
    keep_all = pd.Series(True, index=opps.index)
    rows.append({"controller": "keep all (observed)", "setting": "", **scorer.score(keep_all)})
    baseline = pd.read_parquet(rp.PROJECT_ROOT / "data" / "processed" / f"baseline_test_predictions_{TARGET}.parquet")
    expected = scorer.score_metrics(baseline["y_true"].to_numpy(), baseline["y_score"].to_numpy())["roc_auc"]
    assert abs(rows[0]["roc_auc"] - expected) < 1e-9, (rows[0]["roc_auc"], expected)
    rows.append({"controller": "keep none", "setting": "", **scorer.score(pd.Series(False, index=opps.index))})
    for months, days in [("6-month", 182), ("12-month", 365)]:
        rows.append({"controller": "fixed interval", "setting": months, **scorer.score(fixed_interval_keep(opps, days))})

    for tau in TAUS:
        keep = threshold_policy(opps, scorer.risk_fn, tau)
        rows.append({"controller": "threshold", "setting": f"tau={tau}", **scorer.score(keep)})
        randoms = pd.DataFrame([scorer.score(random_keep(opps, int(keep.sum()), seed)) for seed in RANDOM_SEEDS])
        row = {"controller": "random (matched)", "setting": f"matched to tau={tau}", **randoms.mean().to_dict()}
        row.update({f"{m}_sd": randoms[m].std() for m in ["roc_auc", "brier"]})
        rows.append(row)
        print(f"  tau {tau}: kept {int(keep.sum()):,} | threshold AUROC {rows[-2]['roc_auc']:.4f} "
              f"| random {row['roc_auc']:.4f} +/- {row['roc_auc_sd']:.4f} | {time.perf_counter() - start:.0f}s")

    table = pd.DataFrame(rows)
    rp.save_metrics(table, rp.TABLES_DIR / "scheduling_frontier_masking.csv")
    plot_frontier(table)
    show = ["controller", "setting", "share_of_observed", "spend_per_patient_year", "roc_auc", "brier",
            "keep_ratio_age_<50", "keep_ratio_age_80+"]
    print(table[show].round(4).to_string(index=False))

    log = table.assign(target=TARGET, model=table["controller"] + " " + table["setting"].astype(str))
    write_results(log, task="14", seed=rp.RANDOM_STATE,
                  metrics=["share_of_observed", "spend_per_patient_year", "roc_auc", "brier"],
                  notes="masking evaluation on the test split; calibrated XGBoost; random = mean of 5 seeds")
    print(f"total {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
