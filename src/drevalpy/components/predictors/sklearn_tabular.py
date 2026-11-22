"""Shared helpers for scikit-learn tabular predictors."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from drevalpy.components.hyperparameter_space import is_fixed
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor


class SklearnTabularPredictor(MatrixPredictor):
    """Fit a scikit-learn estimator on available sample and drug features.

    Estimator settings that are not tuned are ``{"type": "fixed"}`` entries of the
    hyperparameter space, so ``self._hyperparameters`` holds every constructor argument and
    :meth:`_estimator_params` hands them to the estimator.
    """

    _estimator: Any = None

    @abstractmethod
    def _make_estimator(self) -> Any:
        """Return an unfitted sklearn-compatible estimator."""

    def _estimator_params(self) -> dict[str, Any]:
        """Resolve every constructor argument of the estimator from the hyperparameters.

        :returns: Hyperparameter values by name, tuned ones coerced to the type of their default.
            ``None`` stays ``None``, such as an unbounded ``max_depth``.
        """
        space = self.get_hyperparameter_space()
        values = self._hyperparameters
        return {
            name: values[name] if is_fixed(spec) or values[name] is None else type(spec["default"])(values[name])
            for name, spec in space.items()
        }

    def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) == 0:
            self._estimator = None
            return
        self._estimator = self._make_estimator()
        self._estimator.fit(x, y)

    def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
        if self._estimator is None:
            return np.full(len(x), np.nan, dtype=np.float64)
        return np.asarray(self._estimator.predict(x), dtype=np.float64)
