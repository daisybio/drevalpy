"""Global mean naive predictor."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.predictors.naive._state_mixin import MeanEffectsStateMixin
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.enums.prediction_mode import PredictionMode


@register(
    "naiveMean",
    tags=("baseline",),
    description="Predict the global mean response.",
    sample_input_format=FeatureFormat.NUMERIC_MATRIX,
    drug_input_format=FeatureFormat.NUMERIC_MATRIX,
)
class NaiveMeanPredictor(MeanEffectsStateMixin, FeatureFreePredictor):
    """Naive mean predictor component."""

    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit on training data.

        :param batch: Training batch; only its responses are used.
        :raises RuntimeError: If ``batch.response`` is None.
        """
        if batch.response is None:
            raise RuntimeError("batch.response is required for fit")
        self._dataset_mean = float(np.mean(batch.response))

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
