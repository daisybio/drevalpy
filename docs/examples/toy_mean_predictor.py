"""Feature-free predictor example: predict the training mean.

``FeatureFreePredictor`` declares ``Inputs.none()`` on both sides, so it sees pair
identifiers and responses only, and config validation rejects a model that gives it
a sample or drug featurizer.
"""

from __future__ import annotations

import numpy as np

from drevalpy.plugin import (
    FeatureFreePredictor,
    ModelInputBatch,
    register_predictor,
)


@register_predictor("toyMean", description="Predict the mean training response for every pair.")
class ToyMeanPredictor(FeatureFreePredictor):
    """The simplest possible baseline."""

    def __init__(self, hyperparameters: dict[str, object] | None = None) -> None:
        """Create an untrained predictor.

        :param hyperparameters: Values merged onto the defaults of the space. This
            predictor has none, but the signature is part of the interface.
        """
        super().__init__(hyperparameters)
        self._mean: float | None = None

    def _fit(self, batch: ModelInputBatch) -> None:
        """Store the mean response.

        ``Predictor.fit`` has already rejected a batch without responses and
        dropped pairs whose features are NaN, so ``_fit`` never has to.
        """
        self._mean = float(np.mean(batch.response))

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Return the stored mean once per pair."""
        if self._mean is None:
            msg = "ToyMeanPredictor must be fitted before predicting"
            raise RuntimeError(msg)
        return np.full(batch.n_pairs, self._mean, dtype=np.float64)
