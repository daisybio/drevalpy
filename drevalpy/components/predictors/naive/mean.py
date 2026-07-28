"""Global mean naive predictor."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.feature_free import FeatureFreePredictor
from drevalpy.components.registry import register_predictor
from drevalpy.components.state_helpers import state_float
from drevalpy.models.config import PredictionMode


@register_predictor(
    "naiveMean",
    tags=("baseline",),
    description="Predict the global mean response.",
)
class NaiveMeanPredictor(FeatureFreePredictor):
    """Naive mean predictor component."""

    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        super().__init__(hyperparameters)
        self._dataset_mean: float | None = None

    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "Naive predictors require response values during fit"
            raise ValueError(msg)
        self._dataset_mean = float(np.mean(batch.response))

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        if self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        return np.full(batch.n_pairs, self._dataset_mean, dtype=np.float64)

    def get_state(self) -> dict[str, object]:
        if self._dataset_mean is None:
            return {}
        return {"dataset_mean": self._dataset_mean}

    def set_state(self, state: dict[str, object]) -> None:
        mean = state_float(state, "dataset_mean")
        if mean is not None:
            self._dataset_mean = mean

    def is_fitted(self) -> bool:
        return self._dataset_mean is not None
