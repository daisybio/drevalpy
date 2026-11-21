"""Predictors that consume flattened dense feature matrices."""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import numpy as np

from drevalpy.components.predictors.abstract.base import Predictor, fit_response
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


class MatrixPredictor(Predictor):
    """Predictor that flattens ``ModelInputBatch`` into one design matrix."""

    sample_inputs: ClassVar[Inputs] = Inputs.matrix()
    drug_inputs: ClassVar[Inputs] = Inputs.matrix()

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit on a dense pair-level design matrix built from *batch*.

        :param batch: Featurized pairs with training responses.
        """
        self._fit_matrix(batch.to_feature_matrix(), fit_response(batch))

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
