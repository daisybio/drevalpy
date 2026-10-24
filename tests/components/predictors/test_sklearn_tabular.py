"""Tests for the shared scikit-learn tabular predictor base."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.sklearn_models import ElasticNetPredictor, RidgePredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.types.enums.prediction_mode import PredictionMode
from tests.components.predictors._helpers import neural_batch


def test_defaults_and_hyperparameter_merging() -> None:
    assert RidgePredictor.supported_modes == frozenset({PredictionMode.REGRESSION})
    assert RidgePredictor()._mode is PredictionMode.REGRESSION
    assert RidgePredictor().is_fitted() is False
    predictor = ElasticNetPredictor(hyperparameters={"max_iter": 7, "alpha": 0.5})

    assert predictor._h["max_iter"] == 7
    assert predictor._h["tol"] == pytest.approx(1e-4)
    assert predictor._h["alpha"] == pytest.approx(0.5)


def test_an_empty_design_matrix_leaves_the_estimator_unset_and_predicts_nan() -> None:
    predictor = RidgePredictor()

    predictor._fit_matrix(np.empty((0, 3)), np.empty(0))
    preds = predictor._predict_matrix(np.zeros((3, 2)))

    assert predictor._estimator is None
    assert predictor.is_fitted() is False
    assert preds.shape == (3,)
    assert np.isnan(preds).all()
    assert preds.dtype == np.float64


def test_state_round_trip_reproduces_predictions() -> None:
    predictor = RidgePredictor(hyperparameters={"alpha": 0.25})
    predictor.fit(neural_batch())
    expected = predictor.predict(neural_batch())
    state = predictor.get_state()
    assert set(state) == {"estimator", "hyperparameters", "mode"}
    assert state["mode"] == PredictionMode.REGRESSION.value

    restored = RidgePredictor()
    restored.set_state(state)

    assert restored.is_fitted()
    assert restored._h["alpha"] == pytest.approx(0.25)
    np.testing.assert_allclose(restored.predict(neural_batch()), expected)
    state["mode"] = PredictionMode.REGRESSION
    instance_restored = RidgePredictor()
    instance_restored.set_state(state)
    assert instance_restored._mode is PredictionMode.REGRESSION


@pytest.mark.parametrize(
    ("entry", "value", "message"),
    [
        ("estimator", None, "missing a fitted estimator"),
        ("hyperparameters", {}, "missing hyperparameters"),
        ("mode", 3, "invalid prediction mode"),
    ],
)
def test_set_state_rejects_malformed_state_and_leaves_the_predictor_untouched(
    entry: str, value: object, message: str
) -> None:
    predictor = RidgePredictor(hyperparameters={"alpha": 0.25})
    predictor.fit(neural_batch())
    expected = predictor.predict(neural_batch())
    state = predictor.get_state()
    if value is None:
        del state[entry]
    else:
        state[entry] = value

    with pytest.raises(PredictorStateError, match=message):
        RidgePredictor().set_state(state)
    with pytest.raises(PredictorStateError, match=message):
        predictor.set_state(state)

    assert predictor._h["alpha"] == pytest.approx(0.25)
    np.testing.assert_allclose(predictor.predict(neural_batch()), expected)
