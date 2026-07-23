"""Predictors that consume flattened dense feature matrices."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors._matrix_fit import validate_matrix_fit
from drevalpy.components.predictors.base import Predictor


class MatrixPredictor(Predictor):
    """Predictor that flattens ``ModelInputBatch`` into one design matrix."""

    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        self._hyperparameters = dict(hyperparameters)
        self._input_dims = dict(input_dims)

    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "Matrix predictors require response values during fit"
            raise ValueError(msg)
        x = batch.to_feature_matrix()
        y = np.asarray(batch.response, dtype=np.float64)
        validate_matrix_fit(x, y, n_pairs=batch.n_pairs)
        self._fit_matrix(x, y)

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        return self._predict_matrix(batch.to_feature_matrix())

    @abstractmethod
    def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit on a dense pair-level design matrix."""

    @abstractmethod
    def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
        """Predict from a dense pair-level design matrix."""
