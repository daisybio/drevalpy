"""Matrix predictor example: ridge regression on the flattened batch.

``MatrixPredictor`` implements ``_fit``/``_predict`` for you by calling
``batch.to_feature_matrix()``, so a subclass only sees a dense pair-level design
matrix through ``_fit_matrix``/``_predict_matrix``. Both contracts must be
``numeric_matrix``; registration rejects anything else for this interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from drevalpy.plugin import (
    FeatureFormat,
    LiteratureReference,
    MatrixPredictor,
    register_predictor,
)

#: Optional provenance metadata. It documents where a ported component came from
#: and changes nothing about training, composition or checkpoints.
TOY_RIDGE_REFERENCE = LiteratureReference(
    repo_url="https://github.com/daisybio/drevalpy",
    citation_text="Ridge baseline written for the DrEvalPy extension guide.",
    deviations="Illustrative only; not a port of any published model.",
)


@register_predictor(
    "toyRidge",
    description="Ridge regression on concatenated dense cell-line and drug features.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=TOY_RIDGE_REFERENCE,
)
class ToyRidgePredictor(MatrixPredictor):
    """Wrap :class:`sklearn.linear_model.Ridge`."""

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Create an untrained predictor.

        Args:
            hyperparameters: Overrides merged onto the declared defaults.
        """
        super().__init__(hyperparameters)
        self._estimator: Ridge | None = None

    @classmethod
    def get_hyperparameter_space(cls) -> dict[str, dict[str, Any]]:
        """Declare the search space HPO samples from.

        Every entry needs a ``default``, which is what the predictor uses when
        nothing is tuned; registration rejects a space that omits one.
        """
        return {
            "alpha": {
                "type": "float",
                "low": 1e-4,
                "high": 10.0,
                "log": True,
                "default": 1.0,
            },
        }

    def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train on the dense pair-level design matrix."""
        self._estimator = Ridge(alpha=float(self._hyperparameters["alpha"]))
        self._estimator.fit(x, y)

    def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
        """Score the dense pair-level design matrix."""
        if self._estimator is None:
            msg = "ToyRidgePredictor must be fitted before predicting"
            raise RuntimeError(msg)
        return np.asarray(self._estimator.predict(x), dtype=np.float64)

    def get_state(self) -> dict[str, object]:
        """Return the trained estimator and the hyperparameters it was built with."""
        if self._estimator is None:
            return {}
        return {"estimator": self._estimator, "hyperparameters": dict(self._hyperparameters)}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore the state produced by ``get_state``."""
        estimator = state.get("estimator")
        if estimator is None:
            return
        self._estimator = estimator  # type: ignore[assignment]
        hyperparameters = state.get("hyperparameters")
        if isinstance(hyperparameters, dict):
            self._hyperparameters = dict(hyperparameters)

    def is_fitted(self) -> bool:
        """Report whether ``fit`` has run."""
        return self._estimator is not None
