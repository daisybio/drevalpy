"""Tests for the XGBoost predictor state round trip."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from drevalpy.registry._builtins import register_builtin_components
from tests.components.predictors._helpers import neural_batch


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_xgboost_load_applies_thread_defaults_before_restore() -> None:
    pytest.importorskip("xgboost")
    from drevalpy.components.predictors.xgboost_pred import XGBoostPredictor, _set_xgboost_thread_defaults

    predictor = XGBoostPredictor(hyperparameters={"n_estimators": 5})
    predictor.fit(neural_batch())
    state = predictor.get_state()

    with patch(
        "drevalpy.components.predictors.xgboost_pred._set_xgboost_thread_defaults",
        wraps=_set_xgboost_thread_defaults,
    ) as thread_defaults:
        restored = XGBoostPredictor()
        restored.set_state(state)
        thread_defaults.assert_called_once()
