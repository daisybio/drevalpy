"""Shared helpers for scikit-learn tabular predictors."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np

from drevalpy.components.predictors._matrix_fit import validate_matrix_fit
from drevalpy.components.predictors.matrix import MatrixPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.state_helpers import state_mapping
from drevalpy.models.config import PredictionMode


class SklearnTabularPredictor(MatrixPredictor):
    """Fit a scikit-learn estimator on available cell-line and drug features."""

    # Estimators are regressors only until classifier implementations exist.
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self) -> None:
        self._h: dict[str, Any] = {}
        self._mode: PredictionMode = PredictionMode.REGRESSION
        self._estimator: Any = None

    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        super().build(hyperparameters, input_dims)
        merged = {**self.get_default_hyperparameters(), **hyperparameters}
        non_tunable = getattr(self, "non_tunable_hyperparameters", None)
        if isinstance(non_tunable, dict):
            merged = {**non_tunable, **merged}
        self._h = merged
        self._mode = PredictionMode(hyperparameters.get("prediction_mode", PredictionMode.REGRESSION))
        self._estimator = None

    @abstractmethod
    def _make_estimator(self) -> Any:
        """Return an unfitted sklearn-compatible estimator."""

    def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        validate_matrix_fit(x, y_arr, n_pairs=len(y_arr))
        if len(x) == 0:
            self._estimator = None
            return
        self._estimator = self._make_estimator()
        self._estimator.fit(x, y_arr)

    def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
        if self._estimator is None:
            return np.full(len(x), np.nan, dtype=np.float64)
        return np.asarray(self._estimator.predict(x), dtype=np.float64)

    def get_state(self) -> dict[str, object]:
        return {
            "estimator": self._estimator,
            "hyperparameters": dict(self._h),
            "mode": self._mode.value,
        }

    def set_state(self, state: dict[str, object]) -> None:
        estimator = state.get("estimator")
        if estimator is None:
            msg = f"{self.__class__.__name__} state is missing a fitted estimator"
            raise PredictorStateError(msg)
        hyperparameters = state_mapping(state, "hyperparameters")
        if not hyperparameters:
            msg = f"{self.__class__.__name__} state is missing hyperparameters"
            raise PredictorStateError(msg)
        self._estimator = estimator
        self._h = {str(key): value for key, value in hyperparameters.items()}
        mode = state.get("mode", PredictionMode.REGRESSION)
        if isinstance(mode, str):
            self._mode = PredictionMode(mode)
        elif isinstance(mode, PredictionMode):
            self._mode = mode
        else:
            msg = f"{self.__class__.__name__} state has an invalid prediction mode"
            raise PredictorStateError(msg)

    def is_fitted(self) -> bool:
        return self._estimator is not None
