"""Shared per-drug routing for scikit-learn matrix predictors."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors._matrix_fit import validate_matrix_fit
from drevalpy.components.predictors.single_drug_routing import (
    iter_drug_masks,
    require_known_training_keys,
    routing_keys,
)
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.state_helpers import state_mapping
from drevalpy.models.config import ModelScope, PredictionMode


class SingleDrugSklearnPredictor(SklearnTabularPredictor):
    """Fit one estimator per drug, using drug identity only for routing."""

    supported_scopes: ClassVar[frozenset[ModelScope]] = frozenset({ModelScope.SINGLE_DRUG})
    routing_drug_featurizer: ClassVar[str] = "identity"

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize instance state.

        :param hyperparameters: hyperparameters.
        """
        super().__init__(hyperparameters)
        self._estimators: dict[str, Any] = {}

    @staticmethod
    def _cell_line_matrix(batch: ModelInputBatch) -> np.ndarray:
        if batch.cell_line_features.size == 0:
            return np.empty((batch.n_pairs, 0), dtype=np.float32)
        return batch.cell_line_features[batch.cell_line_pair_idx]

    def fit(self, batch: ModelInputBatch) -> None:
        """Fit on training data.

        :param batch: batch.
        :raises ValueError: Raised on invalid input.
        """
        if batch.response is None:
            msg = "Single-drug matrix predictors require response values during fit"
            raise ValueError(msg)
        x = self._cell_line_matrix(batch)
        y = np.asarray(batch.response, dtype=np.float64).ravel()
        validate_matrix_fit(x, y, n_pairs=batch.n_pairs)
        keys = routing_keys(batch)
        require_known_training_keys(keys)

        self._estimators = {}
        for drug_id, mask in iter_drug_masks(batch):
            estimator = self._make_estimator()
            estimator.fit(x[mask], y[mask])
            self._estimators[drug_id] = estimator

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param batch: batch.
        :returns: Result.
        """
        x = self._cell_line_matrix(batch)
        keys = routing_keys(batch)
        predictions = np.full(batch.n_pairs, np.nan, dtype=np.float64)
        for drug_id in np.unique(keys):
            estimator = self._estimators.get(str(drug_id))
            if drug_id == "" or estimator is None:
                continue
            mask = keys == drug_id
            predictions[mask] = np.asarray(estimator.predict(x[mask]), dtype=np.float64)
        return predictions

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        return {
            "estimators": dict(self._estimators),
            "hyperparameters": dict(self._h),
            "mode": self._mode.value,
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        :raises PredictorStateError: Raised on invalid input.
        """
        estimators = state_mapping(state, "estimators")
        if not estimators:
            msg = f"{self.__class__.__name__} state is missing fitted per-drug estimators"
            raise PredictorStateError(msg)
        hyperparameters = state_mapping(state, "hyperparameters")
        if not hyperparameters:
            msg = f"{self.__class__.__name__} state is missing hyperparameters"
            raise PredictorStateError(msg)
        self._estimators = {str(key): value for key, value in estimators.items()}
        self._h = {str(key): value for key, value in hyperparameters.items()}
        self._hyperparameters = dict(self._h)
        mode = state.get("mode", PredictionMode.REGRESSION)
        if isinstance(mode, str):
            self._mode = PredictionMode(mode)
        elif isinstance(mode, PredictionMode):
            self._mode = mode
        else:
            msg = f"{self.__class__.__name__} state has an invalid prediction mode"
            raise PredictorStateError(msg)

    def is_fitted(self) -> bool:
        """Return whether the component has been fit.

        :returns: Result.
        """
        return bool(self._estimators)
