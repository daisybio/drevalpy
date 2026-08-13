"""Feature-free predictor example: predict the training mean.

``FeatureFreePredictor`` sees pair identifiers and responses only. Composition
forbids pairing it with cell-line or drug featurizers, but registration still
wants both contracts, because the composition checker compares them before it
knows the interface.
"""

from __future__ import annotations

import numpy as np

from drevalpy.plugin import (
    FeatureFormat,
    FeatureFreePredictor,
    ModelInputBatch,
    register_predictor,
)


@register_predictor(
    "toyMean",
    description="Predict the mean training response for every pair.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class ToyMeanPredictor(FeatureFreePredictor):
    """The simplest possible baseline."""

    def __init__(self, hyperparameters: dict[str, object] | None = None) -> None:
        """Create an untrained predictor.

        Args:
            hyperparameters: Overrides merged onto the declared defaults. This
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

    def get_state(self) -> dict[str, object]:
        """Return the fitted mean for checkpoint persistence."""
        return {} if self._mean is None else {"mean": self._mean}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore the fitted mean produced by ``get_state``."""
        mean = state.get("mean")
        if mean is not None:
            self._mean = float(mean)  # type: ignore[arg-type]

    def is_fitted(self) -> bool:
        """Report whether ``fit`` has run."""
        return self._mean is not None
