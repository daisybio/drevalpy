"""The non-finite check on the final test predictions of :func:`drevalpy.execution._fit.fit`."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import drevalpy
from drevalpy.errors import NaNPredictionError
from drevalpy.models import DRPModel, ModelSpec

FIT_MODULE = importlib.import_module("drevalpy.execution._fit")


def _predictions(n_pairs: int, n_nan: int) -> np.ndarray:
    predictions = np.arange(n_pairs, dtype=float)
    predictions[:n_nan] = np.nan
    return predictions


class TestCheckNonFinitePredictions:
    def test_finite_predictions_pass_without_a_warning(self, caplog) -> None:
        with caplog.at_level("WARNING"):
            FIT_MODULE._check_non_finite_predictions("Model", _predictions(10, 0), np.zeros(10))

        assert caplog.messages == []

    def test_one_nan_prediction_is_warned_about(self, caplog) -> None:
        with caplog.at_level("WARNING"):
            FIT_MODULE._check_non_finite_predictions("Model", _predictions(100, 1), np.zeros(100))

        assert caplog.messages == ["Model: 1 of 100 test predictions are NaN or infinite and are excluded from scoring"]

    def test_infinite_predictions_count_like_nan(self, caplog) -> None:
        predictions = _predictions(100, 11)
        predictions[11:22] = [np.inf, -np.inf] * 5 + [np.inf]

        with (
            caplog.at_level("WARNING"),
            pytest.raises(NaNPredictionError, match="22 of 100 test predictions are NaN or"),
        ):
            FIT_MODULE._check_non_finite_predictions("Model", predictions, np.zeros(100))

    def test_exactly_the_maximum_share_is_still_scored(self) -> None:
        assert FIT_MODULE.MAX_NAN_PREDICTION_SHARE == 0.2

        FIT_MODULE._check_non_finite_predictions("Model", _predictions(100, 20), np.zeros(100))

    def test_more_than_the_maximum_share_raises(self, caplog) -> None:
        with caplog.at_level("WARNING"), pytest.raises(NaNPredictionError, match="21 of 100 test predictions are NaN"):
            FIT_MODULE._check_non_finite_predictions("Model", _predictions(100, 21), np.zeros(100))

        assert len(caplog.messages) == 1

    def test_pairs_without_a_finite_response_are_not_counted(self, caplog) -> None:
        ground_truth = np.zeros(10)
        ground_truth[:5] = np.nan

        with caplog.at_level("WARNING"):
            FIT_MODULE._check_non_finite_predictions("Model", _predictions(10, 6), ground_truth)

        assert caplog.messages == ["Model: 1 of 5 test predictions are NaN or infinite and are excluded from scoring"]


class _NaNForBadModel(DRPModel):
    """A ``DRPModel`` that predicts NaN for every test pair when its spec is named ``Bad``."""

    def predict(self, dataset, scope):
        predictions = super().predict(dataset, scope)
        return np.full_like(predictions, np.nan) if self.name == "Bad" else predictions


@pytest.mark.slow
def test_run_records_a_nan_prediction_error_as_a_failure(synthetic_dataset, monkeypatch) -> None:
    monkeypatch.setattr(FIT_MODULE, "DRPModel", _NaNForBadModel)
    config = ModelSpec.from_name("NaivePredictor").config
    models = [ModelSpec("Bad", config), ModelSpec("Good", config)]

    result = drevalpy.run(models, synthetic_dataset, "LSO", hyperparameter_tuning=False, n_splits=2)

    assert result.model_names == ["Good"]
    assert [(f.identity.model_name, f.error_type) for f in result.failures] == [
        ("Bad", "NaNPredictionError"),
        ("Bad", "NaNPredictionError"),
    ]
