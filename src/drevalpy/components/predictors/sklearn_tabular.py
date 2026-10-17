"""Shared helpers for scikit-learn tabular predictors."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np

from drevalpy.components.predictors._state_helpers import state_mapping
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.types.enums.prediction_mode import PredictionMode


class SklearnTabularPredictor(MatrixPredictor):
    """Fit a scikit-learn estimator on available cell-line and drug features."""

    # Estimators are regressors only until classifier implementations exist.
    supported_modes: ClassVar[frozenset[PredictionMode]] = frozenset({PredictionMode.REGRESSION})

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize instance state.

        :param hyperparameters: hyperparameters.
        """
        super().__init__(hyperparameters)
        merged = dict(self._hyperparameters)
        non_tunable = getattr(self, "non_tunable_hyperparameters", None)
        if isinstance(non_tunable, dict):
            merged = {**non_tunable, **merged}
        self._h = merged
        self._mode = PredictionMode(merged.get("prediction_mode", PredictionMode.REGRESSION))
        self._estimator: Any = None

    @abstractmethod
    def _make_estimator(self) -> Any:
        """Return an unfitted sklearn-compatible estimator."""

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

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        return {"estimator": self._estimator, **self._shared_state()}

    def _shared_state(self) -> dict[str, object]:
        """Return the state entries every sklearn predictor carries besides its estimator(s).

        :returns: The resolved hyperparameters and the prediction mode.
        """
        return {"hyperparameters": dict(self._h), "mode": self._mode.value}

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        :raises PredictorStateError: Raised on invalid input.
        """
        estimator = state.get("estimator")
        if estimator is None:
            msg = f"{self.__class__.__name__} state is missing a fitted estimator"
            raise PredictorStateError(msg)
        shared = self._validated_shared_state(state)
        self._estimator = estimator
        self._apply_shared_state(shared)

    def _validated_shared_state(self, state: dict[str, object]) -> tuple[dict[str, Any], PredictionMode]:
        """Read the hyperparameters and prediction mode out of a state mapping.

        Purely a read: a subclass calls this before assigning anything, so an
        unusable state leaves the predictor as it was.

        :param state: Mapping produced by ``get_state``.
        :returns: The resolved hyperparameters and prediction mode.
        :raises PredictorStateError: If the hyperparameters are absent or the
            prediction mode is neither a string nor a :class:`PredictionMode`.
        """
        hyperparameters = state_mapping(state, "hyperparameters")
        if not hyperparameters:
            msg = f"{self.__class__.__name__} state is missing hyperparameters"
            raise PredictorStateError(msg)
        return (
            {str(key): value for key, value in hyperparameters.items()},
            self._restored_mode(state.get("mode", PredictionMode.REGRESSION)),
        )

    def _apply_shared_state(self, shared: tuple[dict[str, Any], PredictionMode]) -> None:
        """Assign what :meth:`_validated_shared_state` read.

        :param shared: The hyperparameters and prediction mode to adopt.
        """
        self._h, self._mode = shared
        self._hyperparameters = dict(self._h)

    def _restored_mode(self, mode: object) -> PredictionMode:
        """Coerce a serialized prediction mode back to the enum.

        :param mode: Value stored under ``"mode"``.
        :returns: The corresponding enum member.
        :raises PredictorStateError: If *mode* is of an unusable type.
        """
        if isinstance(mode, PredictionMode):
            return mode
        if isinstance(mode, str):
            return PredictionMode(mode)
        msg = f"{self.__class__.__name__} state has an invalid prediction mode"
        raise PredictorStateError(msg)

    def is_fitted(self) -> bool:
        """Return whether the component has been fit.

        :returns: Result.
        """
        return self._estimator is not None
