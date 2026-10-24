"""Tests for the LightGBM tabular predictor."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pytest

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors._boosted_trees import BoostedTreesPredictor
from drevalpy.components.predictors.lightgbm_pred import LightGBMPredictor
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry.predictor import get as get_predictor
from tests.components.predictors._helpers import neural_batch

TINY_HPAMS = {"n_estimators": 3, "max_depth": 2, "num_leaves": 3, "n_jobs": 1}


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_registry_name_contracts_and_class_hierarchy() -> None:
    cls = get_predictor("lightgbm")

    assert cls is LightGBMPredictor
    assert cls.sample_contract.format == FeatureFormat.NUMERIC_MATRIX
    assert cls.drug_contract.format == FeatureFormat.NUMERIC_MATRIX
    assert issubclass(LightGBMPredictor, SklearnTabularPredictor)
    assert issubclass(LightGBMPredictor, BoostedTreesPredictor)


def test_make_estimator_returns_an_unfitted_regressor_with_defaults_and_forwarded_hyperparameters() -> None:
    estimator = LightGBMPredictor(hyperparameters=TINY_HPAMS)._make_estimator()
    assert isinstance(estimator, lgb.LGBMRegressor)
    assert not hasattr(estimator, "booster_")

    default = LightGBMPredictor()._make_estimator()
    assert (default.n_estimators, default.max_depth, default.num_leaves, default.random_state) == (100, 6, 31, 42)
    assert default.learning_rate == pytest.approx(0.1)
    assert default.colsample_bytree == pytest.approx(0.6)
    assert default.get_params()["verbosity"] == -1

    forwarded = LightGBMPredictor(
        hyperparameters={"n_estimators": 7, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 5}
    )._make_estimator()
    assert (forwarded.n_estimators, forwarded.max_depth, forwarded.num_leaves) == (7, 3, 5)
    assert forwarded.learning_rate == pytest.approx(0.05)


def test_hyperparameter_space_is_bounded_and_agrees_with_the_estimator() -> None:
    space = LightGBMPredictor.get_hyperparameter_space()
    estimator = LightGBMPredictor()._make_estimator()

    assert set(space) == {
        "n_estimators",
        "learning_rate",
        "max_depth",
        "num_leaves",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    }
    assert all(spec["low"] <= spec["default"] <= spec["high"] for spec in space.values())
    assert space["n_estimators"]["default"] == estimator.n_estimators
    assert space["num_leaves"]["default"] == estimator.num_leaves
    assert space["learning_rate"]["log"] is True


def test_fit_predict_and_state_round_trip_reproduce_predictions() -> None:
    predictor = LightGBMPredictor(hyperparameters=TINY_HPAMS)
    assert predictor.is_fitted() is False

    predictor.fit(neural_batch())
    expected = predictor.predict(neural_batch())

    assert predictor.is_fitted() is True
    assert expected.shape == (4,)
    assert np.isfinite(expected).all()
    restored = LightGBMPredictor()
    restored.set_state(predictor.get_state())
    assert restored.is_fitted() is True
    np.testing.assert_allclose(restored.predict(neural_batch()), expected)


def test_set_state_rejects_a_missing_estimator() -> None:
    with pytest.raises(PredictorStateError):
        LightGBMPredictor().set_state({"hyperparameters": dict(TINY_HPAMS), "mode": "regression"})


def test_empty_training_matrix_leaves_it_unfitted() -> None:
    predictor = LightGBMPredictor(hyperparameters=TINY_HPAMS)

    predictor._fit_matrix(np.empty((0, 3)), np.empty(0))

    assert predictor.is_fitted() is False
    assert np.isnan(predictor._predict_matrix(np.zeros((2, 3)))).all()
