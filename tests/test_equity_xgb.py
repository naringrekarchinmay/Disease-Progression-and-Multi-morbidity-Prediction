"""Tests for the interval helper of the XGBoost Table 3 (research/equity_xgb.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.equity_xgb import ratio_intervals


def test_intervals_bracket_a_fair_signal_and_exclude_one_aimed_at_a_group():
    rng = np.random.default_rng(0)
    n = 4000
    group = rng.choice(["a", "b"], n)
    fair = pd.DataFrame({"g": group, "s": rng.random(n)})
    ci = ratio_intervals(fair, "g", "s", n_boot=100).set_index(["budget", "group"])
    assert ((ci["ratio_ci_low"] < 1) & (ci["ratio_ci_high"] > 1)).all()
    aimed = fair.assign(s=fair["s"] + (fair["g"] == "a"))  # group a always ranks first
    ci = ratio_intervals(aimed, "g", "s", n_boot=100).set_index(["budget", "group"])
    assert ci.loc[(0.25, "a"), "ratio_ci_low"] > 1.5 and ci.loc[(0.25, "b"), "ratio_ci_high"] < 0.5
