"""Base helpers for predictors that consume named feature blocks."""

from __future__ import annotations

from typing import Any

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.base import Predictor


class BlockPredictor(Predictor):
    """Predictor that reads featurizer outputs from ``ModelInputBatch`` blocks."""

    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        self._hyperparameters = dict(hyperparameters)
        self._input_dims = dict(input_dims)

    def fit(self, batch: ModelInputBatch) -> None:
        """Fit on a featurized predictor input batch."""

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict on a featurized predictor input batch."""
        return np.full(batch.n_pairs, np.nan, dtype=np.float64)


StructuredPredictor = BlockPredictor
