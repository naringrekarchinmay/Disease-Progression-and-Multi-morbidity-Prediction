"""Tests for fitted Q-iteration (research/simulator_fqi.py, ISEM 735 W9-2).

The regressor is pluggable, so these tests use scikit-learn trees; XGBoost is only used in main.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research import simulator_fqi as fqi
from research import simulator_mdp as mdp
from research import simulator_sweep as sweep
from research.cost_model import SIMULATOR_MEASUREMENT_COST
from tests.test_simulator import params, source  # noqa: F401  (fixtures)

C = SIMULATOR_MEASUREMENT_COST


def tree():
    return DecisionTreeRegressor(max_depth=3, min_samples_leaf=5)


def toy_logged(n: int = 400, quarters: int = 3, seed: int = 0) -> pd.DataFrame:
    """Sick patients lose 3 undiagnosed months a quarter until a measurement finds them (from the next
    quarter on); actions are random, like a behavior policy."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sick, found = int(rng.random() < 0.5), 0
        for q in range(quarters):
            a = int(rng.random() < 0.5)
            after = max(found, a * sick)
            rows.append({"patient_id": f"p{i}", "quarter": q, "month": 3 * q, "sick": sick, "found": found,
                         "action": a, "visits": a, "undiagnosed_months": 3 * sick * (1 - found),
                         "terminal": q == quarters - 1, "next_month": 3 * q + 3, "next_sick": sick,
                         "next_found": after})
            found = after
    return pd.DataFrame(rows)


def test_terminal_q_is_the_immediate_reward():
    logged = toy_logged(quarters=1)
    models = fqi.fit_fqi(logged, lam=100.0, state_cols=["month", "sick", "found"], make_regressor=tree,
                         n_quarters=1)
    states = pd.DataFrame({"month": [0, 0], "sick": [0, 1], "found": [0, 0]})
    q0, q1 = fqi.q_values(models[0], states)
    assert q0 == pytest.approx([0.0, -300.0], abs=1e-6)      # 3 undiagnosed months x $100 for the sick
    assert q1 == pytest.approx([-C, -C - 300.0], abs=1e-6)   # detection takes effect from the next quarter


def test_measuring_the_sick_early_is_learned_when_delay_is_expensive():
    logged = toy_logged(quarters=3)
    cols = ["month", "sick", "found"]
    expensive = fqi.fit_fqi(logged, lam=400.0, state_cols=cols, make_regressor=tree, n_quarters=3)
    cheap = fqi.fit_fqi(logged, lam=5.0, state_cols=cols, make_regressor=tree, n_quarters=3)
    states = pd.DataFrame({"month": [0, 0], "sick": [1, 0], "found": [0, 0]})
    q0, q1 = fqi.q_values(expensive[0], states)
    assert q1[0] > q0[0]          # measure the sick patient at $400 per month
    assert q1[1] < q0[1]          # never the healthy one
    q0, q1 = fqi.q_values(cheap[0], states)
    assert q1[0] < q0[0]          # at $5 a month the measurement is not worth it


def test_greedy_policy_uses_the_model_for_the_current_quarter():
    logged = toy_logged(quarters=3)
    cols = ["month", "sick", "found"]
    models = fqi.fit_fqi(logged, lam=400.0, state_cols=cols, make_regressor=tree, n_quarters=3)
    policy = fqi.greedy_policy(models, cols)
    act, p = policy(pd.DataFrame({"month": [0, 0], "sick": [1, 0], "found": [0, 0]}))
    assert act.tolist() == [True, False] and p.tolist() == [1.0, 0.0]
    act, _ = policy(pd.DataFrame({"month": [6], "sick": [1], "found": [1]}))
    assert act.tolist() == [False]  # already found: nothing left to gain


def test_oracle_measures_once_at_the_first_decision_after_each_onset(source, params):  # noqa: F811
    cohort = sweep.cohort_truth(source, params, n_patients=60, seed=3)
    visits = fqi.oracle_visits(cohort)
    months = visits.assign(m=lambda v: (v["test_date"].dt.year - 2018) * 12 + v["test_date"].dt.month - 1)
    assert (months["m"] % 3 == 0).all()
    onsets = cohort.truth[cohort.truth["onset_month"] < 84].copy()
    onsets["first"] = (np.ceil(onsets["onset_month"].clip(lower=0) / 3) * 3).astype(int)
    expected = onsets[onsets["first"] < 84].groupby("patient_id")["first"].apply(lambda s: sorted(set(s)))
    got = months.groupby("patient_id")["m"].apply(sorted)
    assert got.to_dict() == expected.to_dict()


def test_reward_uses_the_mdp_definition():
    df = pd.DataFrame({"visits": [1, 0], "undiagnosed_months": [2, 3]})
    assert fqi.reward(df, 200.0).tolist() == pytest.approx([-C - 400.0, -600.0])
    assert mdp.HEADLINE_LAMBDA == 200.0 and 400.0 in mdp.LAMBDAS
