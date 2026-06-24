"""Shared helpers for scikit-learn tabular predictors."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np

from drevalpy.components.pair_context import PairContext
from drevalpy.components.predictors.base import Predictor
from drevalpy.components.state_helpers import state_mapping
from drevalpy.models.config import PredictionMode


class SklearnTabularPredictor(Predictor):
    """Fit a scikit-learn estimator on concatenated cell-line and drug features."""

    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset(PredictionMode)

    def __init__(self) -> None:
        self._h: dict[str, Any] = {}
        self._mode: PredictionMode = PredictionMode.REGRESSION
        self._estimator: Any = None

    def build(self, hyperparameters: dict[str, Any], input_dims: dict[str, Any]) -> None:
        _ = input_dims
        self._h = hyperparameters
        self._mode = PredictionMode(hyperparameters.get("prediction_mode", PredictionMode.REGRESSION))
        self._estimator = None

    @abstractmethod
    def _make_estimator(self) -> Any:
        """Return an unfitted sklearn-compatible estimator."""

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> None:
        _ = pair_context
        if len(x) == 0:
            self._estimator = None
            return
        self._estimator = self._make_estimator()
        self._estimator.fit(x, np.asarray(y, dtype=np.float64).ravel())

    def predict(
        self,
        x: np.ndarray,
        *,
        pair_context: PairContext | None = None,
    ) -> np.ndarray:
        _ = pair_context
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
        self._estimator = state.get("estimator")
        self._h = {str(key): value for key, value in state_mapping(state, "hyperparameters").items()}
        mode = state.get("mode", PredictionMode.REGRESSION)
        if isinstance(mode, str):
            self._mode = PredictionMode(mode)
        elif isinstance(mode, PredictionMode):
            self._mode = mode

    def is_fitted(self) -> bool:
        return self._estimator is not None
