"""Serialized state for the naive mean-effect predictors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from drevalpy.components.predictors._state_helpers import state_float
from drevalpy.components.predictors.naive._matrix_means import state_float_matrix, state_float_vector

if TYPE_CHECKING:
    import numpy as np


class MeanEffectsStateMixin:
    """Persist a dataset mean plus zero or more named effect arrays.

    Every naive predictor holds fitted state of exactly that shape, so the three
    persistence methods are written once here and configured per class through
    :attr:`state_effects` and :attr:`state_effects_ndim`. An effect named
    ``"drug_effects"`` is read from and written to the attribute
    ``_drug_effects``.
    """

    #: Effect arrays this predictor holds, in serialization order.
    state_effects: ClassVar[tuple[str, ...]] = ()

    #: Dimensionality of every effect array: 1 for a vector, 2 for a matrix.
    state_effects_ndim: ClassVar[int] = 1

    _dataset_mean: float | None

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize instance state.

        :param hyperparameters: hyperparameters.
        """
        super().__init__(hyperparameters)
        self._dataset_mean = None
        for name in self.state_effects:
            setattr(self, f"_{name}", None)

    def _fitted_arrays(self) -> list[np.ndarray] | None:
        if self._dataset_mean is None:
            return None
        arrays = [getattr(self, f"_{name}") for name in self.state_effects]
        if any(array is None for array in arrays):
            return None
        return arrays

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        arrays = self._fitted_arrays()
        if arrays is None:
            return {}
        state: dict[str, object] = {"dataset_mean": self._dataset_mean}
        for name, array in zip(self.state_effects, arrays, strict=True):
            state[name] = array.tolist()
        return state

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        mean = state_float(state, "dataset_mean")
        if mean is not None:
            self._dataset_mean = mean
        restore = state_float_matrix if self.state_effects_ndim == 2 else state_float_vector
        for name in self.state_effects:
            array = restore(state, name)
            if array is not None:
                setattr(self, f"_{name}", array)

    def is_fitted(self) -> bool:
        """Return whether the component has been fit.

        :returns: Result.
        """
        return self._fitted_arrays() is not None
