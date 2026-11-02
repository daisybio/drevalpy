"""Matrix predictor example: ridge regression on the flattened batch.

``MatrixPredictor`` declares ``Inputs.matrix()`` on both sides and implements
``_fit``/``_predict`` for you by calling ``batch.to_feature_matrix()``, so a subclass
only sees a dense pair-level design matrix through ``_fit_matrix``/``_predict_matrix``.
Config validation rejects a featurizer whose output is not numeric.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from drevalpy.plugin import (
    LiteratureReference,
    MatrixPredictor,
    register_predictor,
)

#: Optional provenance metadata. It documents where a ported component came from
#: and changes nothing about training, composition or saved models.
TOY_RIDGE_REFERENCE = LiteratureReference(
    repo_url="https://github.com/daisybio/drevalpy",
    citation_text="Ridge baseline written for the DrEvalPy extension guide.",
    deviations="Illustrative only; not a port of any published model.",
)


@register_predictor(
    "toyRidge",
    description="Ridge regression on concatenated dense sample and drug features.",
    reference=TOY_RIDGE_REFERENCE,
    license="MIT",
)
class ToyRidgePredictor(MatrixPredictor):
    """Wrap :class:`sklearn.linear_model.Ridge`."""

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Create an untrained predictor.

        :param hyperparameters: Values merged onto the defaults of the space.
        """
        super().__init__(hyperparameters)
        self._estimator: Ridge | None = None

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Declare the search space HPO draws from.

        Every entry needs a ``default``, which is what the predictor uses when
        nothing is tuned; registration rejects a space that omits one. A
        ``fixed`` entry is a setting HPO never draws.
        """
        return {
            "alpha": {
                "type": "float",
                "low": 1e-4,
                "high": 10.0,
                "log": True,
                "default": 1.0,
            },
            "fit_intercept": {"type": "fixed", "default": True},
        }

    def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train on the dense pair-level design matrix."""
        self._estimator = Ridge(
            alpha=float(self._hyperparameters["alpha"]),
            fit_intercept=bool(self._hyperparameters["fit_intercept"]),
        )
        self._estimator.fit(x, y)

    def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
        """Score the dense pair-level design matrix."""
        if self._estimator is None:
            msg = "ToyRidgePredictor must be fitted before predicting"
            raise RuntimeError(msg)
        return np.asarray(self._estimator.predict(x), dtype=np.float64)
