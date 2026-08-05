"""Predictors that consume flattened dense feature matrices."""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors._matrix_fit import validate_matrix_fit
from drevalpy.components.predictors.base import Predictor


class MatrixPredictor(Predictor):
    """Predictor that flattens ``ModelInputBatch`` into one design matrix."""

    input_interface: ClassVar[str] = "matrix"

    def fit(self, batch: ModelInputBatch) -> None:
        """Fit on a dense pair-level design matrix built from *batch*.

        :param batch: Featurized pairs with training responses.

        :raises ValueError: If responses are missing or the design matrix is invalid.
        """
        if batch.response is None:
            msg = "Matrix predictors require response values during fit"
            raise ValueError(msg)
        x = batch.to_feature_matrix()
        y = np.asarray(batch.response, dtype=np.float64)
        validate_matrix_fit(x, y, n_pairs=batch.n_pairs)
        self._fit_matrix(x, y)

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict from a dense pair-level design matrix built from *batch*.

        :param batch: Featurized pairs to score.

        :returns: One predicted response per pair in *batch*.
        """
        return self._predict_matrix(batch.to_feature_matrix())

    @abstractmethod
    def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit on a dense pair-level design matrix.

        :param x: Pair-level feature matrix.
        :param y: Training responses aligned with *x*.
        """

    @abstractmethod
    def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
        """Predict from a dense pair-level design matrix.

        :param x: Pair-level feature matrix.
        :returns: Predicted responses aligned with rows of *x*.
        """
