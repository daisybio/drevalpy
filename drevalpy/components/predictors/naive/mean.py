"""Global mean naive predictor."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.abstract.feature_free import FeatureFreePredictor
from drevalpy.components.registry import register_predictor
from drevalpy.components.state_helpers import state_float
from drevalpy.models.config import PredictionMode


@register_predictor(
    "naiveMean",
    tags=("baseline",),
    description="Predict the global mean response.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class NaiveMeanPredictor(FeatureFreePredictor):
    """Naive mean predictor component."""

    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize instance state.

        :param hyperparameters: hyperparameters.
        """
        super().__init__(hyperparameters)
        self._dataset_mean: float | None = None

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit on training data.

        :param batch: batch.
        """
        self._dataset_mean = float(np.mean(batch.response))

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param batch: batch.
        :returns: Result.
        :raises RuntimeError: Raised on invalid input.
        """
        if self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        return np.full(batch.n_pairs, self._dataset_mean, dtype=np.float64)

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        if self._dataset_mean is None:
            return {}
        return {"dataset_mean": self._dataset_mean}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        mean = state_float(state, "dataset_mean")
        if mean is not None:
            self._dataset_mean = mean

    def is_fitted(self) -> bool:
        """Return whether the component has been fit.

        :returns: Result.
        """
        return self._dataset_mean is not None
