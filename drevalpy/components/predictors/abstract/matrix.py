"""Predictors that consume flattened dense feature matrices."""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import numpy as np

from drevalpy.components.core.batch.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.abstract.base import Predictor


class MatrixPredictor(Predictor):
    """Predictor that flattens ``ModelInputBatch`` into one design matrix."""

    input_interface: ClassVar[str] = "matrix"

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit on a dense pair-level design matrix built from *batch*.

        :param batch: Featurized pairs with training responses.
        :raises RuntimeError: If batch.response is None.
        """
        x = batch.to_feature_matrix()
        if batch.response is None:
            raise RuntimeError("batch.response is required for fit")
        self._fit_matrix(x, batch.response)

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
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
