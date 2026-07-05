"""Survival analysis for time to multimorbidity progression.

We study how long patients in the "at-risk" cohort (fewer than two baseline
conditions at the 2021-12-31 cutoff) stay free of multimorbidity, where the
event is reaching two distinct conditions. Patients who never reach two
conditions by the 2024-12-31 horizon are right-censored at the horizon.

lifelines is imported lazily inside the plotting/fitting functions so importing
this module never fails when the optional package is missing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CUTOFF = "2021-12-31"
HORIZON = "2024-12-31"

COX_COVARIATES = [
    "baseline_condition_count",
    "age",
    "charlson_index",
    "prior_hospitalization_count",
]


def build_time_to_progression(
    condition_events: pd.DataFrame,
    feature_df: pd.DataFrame,
    cutoff: str = CUTOFF,
    horizon: str = HORIZON,
) -> pd.DataFrame:
    """Return one row per at-risk patient with a progression time and event flag.

    ``duration_days`` counts days from the cutoff to progression (event) or to
    the horizon (censored). Because the at-risk cohort starts with 0 or 1
    baseline conditions, progression needs ``2 - baseline_count`` new distinct
    conditions, so we can find the progression date from the ordered
    first-occurrence dates of post-cutoff conditions.
    """
    origin = pd.Timestamp(cutoff)
    horizon_ts = pd.Timestamp(horizon)
    max_days = (horizon_ts - origin).days

    events = condition_events.copy()
    events["patient_id"] = events["patient_id"].astype(str)
    events["visit_date"] = pd.to_datetime(events["visit_date"], errors="coerce")

    cohort = feature_df.loc[
        feature_df["baseline_condition_count"] < 2,
        ["patient_id", *COX_COVARIATES],
    ].copy()
    cohort["patient_id"] = cohort["patient_id"].astype(str)
    at_risk_ids = set(cohort["patient_id"])

    # The single baseline condition (if any) so we do not recount it as "new".
    baseline = events[(events["visit_date"] <= origin) & events["patient_id"].isin(at_risk_ids)]
    baseline_sets = baseline.groupby("patient_id")["condition"].agg(set)
    baseline_condition = {
        pid: next(iter(conds)) for pid, conds in baseline_sets.items() if len(conds) == 1
    }

    # First time each new condition appears after the cutoff.
    post = events[
        (events["visit_date"] > origin)
        & (events["visit_date"] <= horizon_ts)
        & events["patient_id"].isin(at_risk_ids)
    ]
    first_seen = post.groupby(["patient_id", "condition"])["visit_date"].min().reset_index()

    # Drop a new condition that merely repeats the patient's baseline condition.
    first_seen["baseline_condition"] = first_seen["patient_id"].map(baseline_condition)
    first_seen = first_seen[first_seen["condition"] != first_seen["baseline_condition"]]

    # Rank distinct new conditions by date; progression is the "needed"-th one.
    first_seen = first_seen.sort_values(["patient_id", "visit_date"])
    first_seen["distinct_rank"] = first_seen.groupby("patient_id").cumcount()
    needed = (2 - cohort.set_index("patient_id")["baseline_condition_count"]).astype(int)
    first_seen["needed"] = first_seen["patient_id"].map(needed)
    progression = (
        first_seen[first_seen["distinct_rank"] == first_seen["needed"] - 1]
        .set_index("patient_id")["visit_date"]
    )

    surv = cohort.set_index("patient_id")
    surv["progression_date"] = progression
    surv["event_observed"] = surv["progression_date"].notna().astype(int)
    days_to_event = (surv["progression_date"] - origin).dt.days
    surv["duration_days"] = np.where(surv["event_observed"] == 1, days_to_event, max_days)
    surv["duration_days"] = surv["duration_days"].clip(lower=1, upper=max_days)
    return surv.drop(columns="progression_date").reset_index()


def plot_km_by_condition_group(survival_df: pd.DataFrame, path: str | Path | None = None):
    """Kaplan-Meier progression-free survival stratified by baseline condition count."""
    from lifelines import KaplanMeierFitter

    fig, ax = plt.subplots(figsize=(7, 5))
    kmf = KaplanMeierFitter()
    for count, group in survival_df.groupby("baseline_condition_count"):
        kmf.fit(
            group["duration_days"],
            group["event_observed"],
            label=f"{int(count)} baseline condition(s) (n={len(group):,})",
        )
        kmf.plot_survival_function(ax=ax)
    ax.set_title("Progression-Free Survival by Baseline Condition Count")
    ax.set_xlabel("Days since cutoff (2021-12-31)")
    ax.set_ylabel("Probability of staying below 2 conditions")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig, ax


def fit_cox_model(
    survival_df: pd.DataFrame,
    covariates: list[str] | None = None,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Fit a Cox proportional hazards baseline model and return its summary table."""
    from lifelines import CoxPHFitter

    covariates = covariates or COX_COVARIATES
    model_df = survival_df[["duration_days", "event_observed", *covariates]].dropna()

    cph = CoxPHFitter()
    cph.fit(model_df, duration_col="duration_days", event_col="event_observed")

    summary = cph.summary.copy()
    summary.insert(0, "covariate", summary.index)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path, index=False)
    return summary
