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


def test_xgboost_shares_the_boosting_base_with_lightgbm() -> None:
    assert issubclass(XGBoostPredictor, BoostedTreesPredictor)


def test_xgboost_hyperparameter_space_tunes_three_knobs() -> None:
    assert set(XGBoostPredictor.get_hyperparameter_space()) == {"n_estimators", "max_depth", "learning_rate"}


def test_xgboost_caps_the_tree_depth_below_the_shared_ceiling() -> None:
    assert XGBoostPredictor.get_hyperparameter_space()["max_depth"]["high"] == 8


def test_xgboost_samples_the_learning_rate_uniformly() -> None:
    assert "log" not in XGBoostPredictor.get_hyperparameter_space()["learning_rate"]


def test_xgboost_hyperparameter_space_declares_bounded_specs() -> None:
    space = XGBoostPredictor.get_hyperparameter_space()

    assert all(spec["low"] <= spec["default"] <= spec["high"] for spec in space.values())


def test_xgboost_estimator_params_keep_the_documented_defaults() -> None:
    params = XGBoostPredictor()._estimator_params()

    assert params["n_estimators"] == 100
    assert params["max_depth"] == 6
    assert params["learning_rate"] == pytest.approx(0.1)
    assert params["subsample"] == pytest.approx(1.0)
    assert params["colsample_bytree"] == pytest.approx(1.0)
    assert params["reg_alpha"] == pytest.approx(0.0)
    assert params["random_state"] == 42


def test_xgboost_does_not_pass_lightgbm_only_arguments() -> None:
    params = XGBoostPredictor()._estimator_params()

    assert "num_leaves" not in params
    assert "reg_lambda" not in params


def test_xgboost_make_estimator_forwards_hyperparameters() -> None:
    pytest.importorskip("xgboost")

    estimator = XGBoostPredictor(hyperparameters={"n_estimators": 7, "max_depth": 3})._make_estimator()

    assert estimator.n_estimators == 7
    assert estimator.max_depth == 3


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
