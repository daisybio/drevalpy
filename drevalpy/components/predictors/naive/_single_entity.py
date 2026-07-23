"""Shared base for entity-level naive predictors."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.structured import BlockPredictor
from drevalpy.components.state_helpers import state_float, state_str_dict


class SingleEntityNaivePredictor(BlockPredictor):
    """Predict per-entity means with a global fallback."""

    requires_drug_featurizer: ClassVar[bool] = False

    def __init__(self) -> None:
        self._dataset_mean: float | None = None
        self._entity_means: dict[str, float] = {}

    def _entity_keys(self, batch: ModelInputBatch) -> np.ndarray:
        raise NotImplementedError

    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "Naive predictors require response values during fit"
            raise ValueError(msg)
        y = np.asarray(batch.response, dtype=np.float64)
        self._dataset_mean = float(np.mean(y))
        keys = self._entity_keys(batch)
        for entity in np.unique(keys.astype(str)):
            mask = keys.astype(str) == entity
            self._entity_means[str(entity)] = float(np.mean(y[mask]))

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        if self._dataset_mean is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        keys = self._entity_keys(batch).astype(str)
        return np.array(
            [self._entity_means.get(key, self._dataset_mean) for key in keys],
            dtype=np.float64,
        )

    def _legacy_entity_means_key(self) -> str:
        raise NotImplementedError

    def get_state(self) -> dict[str, object]:
        if self._dataset_mean is None:
            return {}
        return {
            "dataset_mean": self._dataset_mean,
            self._legacy_entity_means_key(): dict(self._entity_means),
        }

    def set_state(self, state: dict[str, object]) -> None:
        mean = state_float(state, "dataset_mean")
        if mean is not None:
            self._dataset_mean = mean
        entity_means = state_str_dict(state, self._legacy_entity_means_key())
        if entity_means:
            self._entity_means = entity_means

    def is_fitted(self) -> bool:
        return self._dataset_mean is not None
