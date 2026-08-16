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
from drevalpy.registry._builtins import ensure_predictor_registered, register_builtin_components
from drevalpy.registry.predictor import get as get_predictor
from tests.components.predictors._helpers import neural_batch

#: ``n_jobs=1`` keeps LightGBM single-threaded: its OpenMP runtime clashes with
#: the one torch loads, which the rest of this suite pulls in.
TINY_HPAMS = {"n_estimators": 3, "max_depth": 2, "num_leaves": 3, "n_jobs": 1}


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_lightgbm_predictor_registry_name() -> None:
    ensure_predictor_registered("lightgbm")

    assert get_predictor("lightgbm") is LightGBMPredictor


def test_lightgbm_predictor_requires_numeric_matrix_contracts() -> None:
    cls = get_predictor("lightgbm")

    assert cls.cell_line_contract.format == FeatureFormat.NUMERIC_MATRIX
    assert cls.drug_contract.format == FeatureFormat.NUMERIC_MATRIX


def test_lightgbm_predictor_reuses_the_sklearn_tabular_lifecycle() -> None:
    assert issubclass(LightGBMPredictor, SklearnTabularPredictor)


def test_lightgbm_predictor_shares_the_boosting_base_with_xgboost() -> None:
    assert issubclass(LightGBMPredictor, BoostedTreesPredictor)


def test_lightgbm_make_estimator_returns_an_unfitted_regressor() -> None:
    estimator = LightGBMPredictor(hyperparameters=TINY_HPAMS)._make_estimator()

    assert isinstance(estimator, lgb.LGBMRegressor)
    assert not hasattr(estimator, "booster_")


def test_lightgbm_make_estimator_forwards_hyperparameters() -> None:
    estimator = LightGBMPredictor(
        hyperparameters={"n_estimators": 7, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 5}
    )._make_estimator()

    assert estimator.n_estimators == 7
    assert estimator.learning_rate == pytest.approx(0.05)
    assert estimator.max_depth == 3
    assert estimator.num_leaves == 5


def test_lightgbm_make_estimator_uses_documented_defaults() -> None:
    estimator = LightGBMPredictor()._make_estimator()

    assert estimator.n_estimators == 100
    assert estimator.learning_rate == pytest.approx(0.1)
    assert estimator.max_depth == 6
    assert estimator.num_leaves == 63
    assert estimator.random_state == 42


def test_lightgbm_make_estimator_silences_training_output() -> None:
    estimator = LightGBMPredictor()._make_estimator()

    assert estimator.get_params()["verbosity"] == -1


def test_lightgbm_hyperparameter_space_defaults_match_the_estimator() -> None:
    space = LightGBMPredictor.get_hyperparameter_space()
    estimator = LightGBMPredictor()._make_estimator()

    assert space["n_estimators"]["default"] == estimator.n_estimators
    assert space["num_leaves"]["default"] == estimator.num_leaves


def test_lightgbm_hyperparameter_space_declares_bounded_specs() -> None:
    space = LightGBMPredictor.get_hyperparameter_space()

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


def test_lightgbm_samples_the_learning_rate_log_uniformly() -> None:
    assert LightGBMPredictor.get_hyperparameter_space()["learning_rate"]["log"] is True


def test_lightgbm_predictor_is_not_fitted_before_fit() -> None:
    assert LightGBMPredictor(hyperparameters=TINY_HPAMS).is_fitted() is False


def test_lightgbm_predictor_fits_and_predicts_one_value_per_pair() -> None:
    predictor = LightGBMPredictor(hyperparameters=TINY_HPAMS)

    predictor.fit(neural_batch())
    preds = predictor.predict(neural_batch())

    assert predictor.is_fitted() is True
    assert preds.shape == (4,)
    assert np.isfinite(preds).all()


def test_lightgbm_predictor_state_round_trip_reproduces_predictions() -> None:
    predictor = LightGBMPredictor(hyperparameters=TINY_HPAMS)
    predictor.fit(neural_batch())
    expected = predictor.predict(neural_batch())

    restored = LightGBMPredictor()
    restored.set_state(predictor.get_state())

    assert restored.is_fitted() is True
    np.testing.assert_allclose(restored.predict(neural_batch()), expected)


def test_lightgbm_predictor_set_state_rejects_a_missing_estimator() -> None:
    with pytest.raises(PredictorStateError):
        LightGBMPredictor().set_state({"hyperparameters": dict(TINY_HPAMS), "mode": "regression"})


def test_lightgbm_predictor_empty_training_matrix_leaves_it_unfitted() -> None:
    predictor = LightGBMPredictor(hyperparameters=TINY_HPAMS)

    predictor._fit_matrix(np.empty((0, 3)), np.empty(0))

    assert predictor.is_fitted() is False
    assert np.isnan(predictor._predict_matrix(np.zeros((2, 3)))).all()
