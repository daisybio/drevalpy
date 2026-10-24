"""Tests for the XGBoost predictor state round trip and hyperparameter wiring."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from drevalpy.components.predictors._boosted_trees import BoostedTreesPredictor
from drevalpy.components.predictors.xgboost_pred import XGBoostPredictor
from drevalpy.registry._builtins import register_builtin_components
from tests.components.predictors._helpers import neural_batch


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_hyperparameter_space_tunes_the_1_6_1_grid_keys_with_bounded_specs() -> None:
    space = XGBoostPredictor.get_hyperparameter_space()

    assert issubclass(XGBoostPredictor, BoostedTreesPredictor)
    assert set(space) == {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    }
    assert all(spec["low"] <= spec["default"] <= spec["high"] for spec in space.values())
    assert space["max_depth"]["low"] <= 10 <= space["max_depth"]["high"]
    assert "log" not in space["learning_rate"]


def test_estimator_params_keep_the_documented_defaults_and_forward_overrides() -> None:
    params = XGBoostPredictor()._estimator_params()

    assert (params["n_estimators"], params["max_depth"], params["random_state"]) == (100, 10, 42)
    assert params["learning_rate"] == pytest.approx(0.1)
    assert params["subsample"] == pytest.approx(0.8)
    assert params["colsample_bytree"] == pytest.approx(0.6)
    assert params["reg_alpha"] == pytest.approx(0.0)
    assert params["reg_lambda"] == pytest.approx(0.1)
    assert "num_leaves" not in params
    pytest.importorskip("xgboost")
    estimator = XGBoostPredictor(hyperparameters={"n_estimators": 7, "max_depth": 3})._make_estimator()
    assert (estimator.n_estimators, estimator.max_depth) == (7, 3)


def test_xgboost_load_applies_thread_defaults_before_restore() -> None:
    pytest.importorskip("xgboost")
    from drevalpy.components.predictors.xgboost_pred import _set_xgboost_thread_defaults

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
