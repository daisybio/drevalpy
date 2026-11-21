"""Global mean naive predictor."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.abstract.base import get_response
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.registry.predictor import register
from drevalpy.types.data.predictor_input import PredictorInput


@register(
    "naiveMean",
    tags=("baseline",),
    description="Predict the global mean response.",
)
class NaiveMeanPredictor(FeatureFreePredictor):
    """Naive mean predictor component."""

    _dataset_mean: float | None = None

    def _fit(self, data: PredictorInput) -> None:
        """Fit on training data.

        :param data: Training batch; only its responses are used.
        """
        self._dataset_mean = float(np.mean(get_response(data)))

    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param data: Pairs to score.
        :returns: The dataset mean for every pair.
        :raises RuntimeError: If the predictor has not been fitted.
        """
        if self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        return np.full(data.n_pairs, self._dataset_mean, dtype=np.float64)
