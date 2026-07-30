"""Shared per-drug routing for scikit-learn matrix predictors."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors._matrix_fit import validate_matrix_fit
from drevalpy.components.predictors.sklearn_tabular import SklearnTabularPredictor
from drevalpy.components.predictors.state_errors import PredictorStateError
from drevalpy.components.state_helpers import state_mapping
from drevalpy.models.config import ModelScope, PredictionMode


class SingleDrugSklearnPredictor(SklearnTabularPredictor):
    """Fit one estimator per drug, using drug identity only for routing."""

    supported_scopes: ClassVar[frozenset[ModelScope]] = frozenset({ModelScope.SINGLE_DRUG})
    routing_drug_featurizer: ClassVar[str] = "identity"

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        super().__init__(hyperparameters)
        self._estimators: dict[str, Any] = {}

    @staticmethod
    def _cell_line_matrix(batch: ModelInputBatch) -> np.ndarray:
        if batch.cell_line_features.size == 0:
            return np.empty((batch.n_pairs, 0), dtype=np.float32)
        return batch.cell_line_features[batch.cell_line_pair_idx]

    @staticmethod
    def _routing_keys(batch: ModelInputBatch) -> np.ndarray:
        identity_block = batch.drug_blocks.get("identity")
        categories_block = batch.drug_blocks.get("identity_categories")
        if identity_block is None or categories_block is None or batch.drug_pair_idx is None:
            msg = "Single-drug predictors require drug identity features for per-drug routing"
            raise ValueError(msg)

        identity_matrix = np.asarray(identity_block.values)
        category_ids = np.asarray(categories_block.values, dtype=str).reshape(-1)
        if identity_matrix.ndim != 2 or identity_matrix.shape[1] != len(category_ids):
            msg = "Drug identity features and identity categories are misaligned"
            raise ValueError(msg)

        pair_identity = identity_matrix[batch.drug_pair_idx]
        known = np.isclose(pair_identity.sum(axis=1), 1.0)
        keys = np.full(batch.n_pairs, "", dtype=object)
        if np.any(known):
            keys[known] = category_ids[np.argmax(pair_identity[known], axis=1)]
        return np.asarray(keys, dtype=str)

    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "Single-drug matrix predictors require response values during fit"
            raise ValueError(msg)
        x = self._cell_line_matrix(batch)
        y = np.asarray(batch.response, dtype=np.float64).ravel()
        validate_matrix_fit(x, y, n_pairs=batch.n_pairs)
        routing_keys = self._routing_keys(batch)
        if np.any(routing_keys == ""):
            msg = "Training pairs contain unknown drug identities"
            raise ValueError(msg)

        self._estimators = {}
        for drug_id in np.unique(routing_keys):
            mask = routing_keys == drug_id
            estimator = self._make_estimator()
            estimator.fit(x[mask], y[mask])
            self._estimators[str(drug_id)] = estimator

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        x = self._cell_line_matrix(batch)
        routing_keys = self._routing_keys(batch)
        predictions = np.full(batch.n_pairs, np.nan, dtype=np.float64)
        for drug_id in np.unique(routing_keys):
            estimator = self._estimators.get(str(drug_id))
            if drug_id == "" or estimator is None:
                continue
            mask = routing_keys == drug_id
            predictions[mask] = np.asarray(estimator.predict(x[mask]), dtype=np.float64)
        return predictions

    def get_state(self) -> dict[str, object]:
        return {
            "estimators": dict(self._estimators),
            "hyperparameters": dict(self._h),
            "mode": self._mode.value,
        }

    def set_state(self, state: dict[str, object]) -> None:
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
        return bool(self._estimators)
