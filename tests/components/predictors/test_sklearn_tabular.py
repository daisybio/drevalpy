"""Tests for the shared scikit-learn tabular predictor base.

Carved out of ``test_sklearn_models.py``, which covers the concrete estimator
zoo. This file covers the base class that ``sklearn_models``, ``xgboost_pred``
and ``lightgbm_pred`` all inherit: hyperparameter merging, the degenerate
empty-matrix path, and the three ``set_state`` rejection branches.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.sklearn_models import ElasticNetPredictor, RidgePredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.types.enums.prediction_mode import PredictionMode
from tests.components.predictors._helpers import neural_batch


def test_sklearn_tabular_supports_regression_only() -> None:
    assert RidgePredictor.supported_modes == frozenset({PredictionMode.REGRESSION})


def test_sklearn_tabular_defaults_to_regression_mode() -> None:
    predictor = RidgePredictor()

    assert predictor._mode is PredictionMode.REGRESSION


def test_sklearn_tabular_merges_non_tunable_hyperparameters_under_explicit_ones() -> None:
    predictor = ElasticNetPredictor(hyperparameters={"max_iter": 7, "alpha": 0.5})

    assert predictor._h["max_iter"] == 7
    assert predictor._h["tol"] == pytest.approx(1e-4)
    assert predictor._h["alpha"] == pytest.approx(0.5)


def test_sklearn_tabular_is_not_fitted_before_fit() -> None:
    assert RidgePredictor().is_fitted() is False


def test_sklearn_tabular_empty_design_matrix_leaves_estimator_unset() -> None:
    predictor = RidgePredictor()

    predictor._fit_matrix(np.empty((0, 3)), np.empty(0))

    assert predictor._estimator is None
    assert predictor.is_fitted() is False


def test_sklearn_tabular_predicts_nan_without_a_fitted_estimator() -> None:
    predictor = RidgePredictor()

    preds = predictor._predict_matrix(np.zeros((3, 2)))

    assert preds.shape == (3,)
    assert np.isnan(preds).all()
    assert preds.dtype == np.float64


def test_sklearn_tabular_state_round_trip_reproduces_predictions() -> None:
    predictor = RidgePredictor(hyperparameters={"alpha": 0.25})
    predictor.fit(neural_batch())
    expected = predictor.predict(neural_batch())

    restored = RidgePredictor()
    restored.set_state(predictor.get_state())

    assert restored.is_fitted()
    assert restored._h["alpha"] == pytest.approx(0.25)
    np.testing.assert_allclose(restored.predict(neural_batch()), expected)


def test_sklearn_tabular_get_state_reports_the_prediction_mode_value() -> None:
    predictor = RidgePredictor()
    predictor.fit(neural_batch())

    assert predictor.get_state()["mode"] == PredictionMode.REGRESSION.value


def test_sklearn_tabular_set_state_accepts_a_prediction_mode_instance() -> None:
    predictor = RidgePredictor()
    predictor.fit(neural_batch())
    state = predictor.get_state()
    state["mode"] = PredictionMode.REGRESSION

    restored = RidgePredictor()
    restored.set_state(state)

    assert restored._mode is PredictionMode.REGRESSION


def test_sklearn_tabular_set_state_rejects_missing_estimator() -> None:
    predictor = RidgePredictor()

    with pytest.raises(PredictorStateError, match="missing a fitted estimator"):
        predictor.set_state({"hyperparameters": {"alpha": 1.0}, "mode": "regression"})


def test_sklearn_tabular_set_state_rejects_missing_hyperparameters() -> None:
    predictor = RidgePredictor()
    predictor.fit(neural_batch())
    state = predictor.get_state()
    state["hyperparameters"] = {}

    with pytest.raises(PredictorStateError, match="missing hyperparameters"):
        RidgePredictor().set_state(state)


def test_sklearn_tabular_set_state_rejects_an_invalid_prediction_mode() -> None:
    predictor = RidgePredictor()
    predictor.fit(neural_batch())
    state = predictor.get_state()
    state["mode"] = 3

    with pytest.raises(PredictorStateError, match="invalid prediction mode"):
        RidgePredictor().set_state(state)
