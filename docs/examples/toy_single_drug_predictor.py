"""Single-drug predictor example: one ridge regression per drug on the sample features.

``SingleDrugPredictor`` declares ``drug_inputs = Inputs.none()`` and the single-drug scope.
It routes the pairs by drug, fits the drugs in worker processes when ``n_jobs`` is above 1,
and predicts NaN for a drug it has no model for. A subclass declares its sample inputs and
implements two hooks, each on a batch that holds the pairs of one drug.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.linear_model import Ridge

from drevalpy.plugin import (
    Inputs,
    ModelInputBatch,
    SingleDrugPredictor,
    register_predictor,
)


def _sample_matrix(batch: ModelInputBatch) -> np.ndarray:
    """Expand the sample features, one row per sample, to one row per pair."""
    return batch.sample.matrix[batch.sample_pair_idx]


@register_predictor("toySingleDrugRidge", description="One ridge regression per drug on dense sample features.")
class ToySingleDrugRidgePredictor(SingleDrugPredictor):
    """Fit :class:`sklearn.linear_model.Ridge` on the pairs of each drug."""

    sample_inputs: ClassVar[Inputs] = Inputs.matrix()

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Declare the penalty that every per-drug model uses."""
        return {"alpha": {"type": "float", "low": 1e-4, "high": 10.0, "log": True, "default": 1.0}}

    def _fit_drug(self, drug_id: str, batch: ModelInputBatch) -> Ridge | None:
        """Fit the model of one drug, or skip a drug with too few pairs.

        This may run in a worker process, so it returns the model instead of storing it on
        ``self``; the base class keeps it for ``_predict_drug``.
        """
        if batch.n_pairs < 2:
            return None
        model = Ridge(alpha=float(self._hyperparameters["alpha"]))
        model.fit(_sample_matrix(batch), batch.response)
        return model

    def _predict_drug(self, drug_id: str, model: Ridge, batch: ModelInputBatch) -> np.ndarray:
        """Score the pairs of one drug with the model ``_fit_drug`` returned for it."""
        return model.predict(_sample_matrix(batch))
