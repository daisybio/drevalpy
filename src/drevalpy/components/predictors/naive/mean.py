"""Global mean naive predictor."""

from __future__ import annotations

import numpy as np

from drevalpy.components.predictors.abstract.base import fit_response
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


@register(
    "naiveMean",
    tags=("baseline",),
    description="Predict the global mean response.",
)
class NaiveMeanPredictor(FeatureFreePredictor):
    """Naive mean predictor component."""

    _dataset_mean: float | None = None

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit on training data.

        :param batch: Training batch; only its responses are used.
        """
        self._dataset_mean = float(np.mean(fit_response(batch)))

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param batch: Pairs to score.
        :returns: The dataset mean for every pair.
        :raises RuntimeError: If the predictor has not been fitted.
        """
        if self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        return np.full(batch.n_pairs, self._dataset_mean, dtype=np.float64)
