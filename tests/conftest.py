"""Shared pytest setup.

PyTorch and XGBoost each bundle their own libomp, and on macOS the two in one process
segfault or hang, whichever loads first. pytest runs every test module in one process and
the torch tests import torch at collection, so XGBoost must never load in that process.
Tests that fit XGBoost run their body in a fresh interpreter (see test_calibration.py);
this guard turns a stray in-process load into a clear failure instead of a segfault.
"""

import sys

import pytest


@pytest.fixture(autouse=True)
def _no_xgboost_in_the_test_process():
    yield
    if "xgboost" in sys.modules:
        pytest.fail("xgboost was loaded into the pytest process; run XGBoost code in a subprocess.")
