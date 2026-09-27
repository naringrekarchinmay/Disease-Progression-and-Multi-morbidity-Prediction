"""Shared pytest setup.

pytest runs every test module in one process. PyTorch and XGBoost each bundle their
own libomp, and on macOS loading torch's first makes XGBoost segfault or hang. Loading
XGBoost here, before any test module imports torch, keeps the order safe for the
small tensors the tests use. Real runs keep the two in separate processes.
"""

import xgboost  # noqa: F401
