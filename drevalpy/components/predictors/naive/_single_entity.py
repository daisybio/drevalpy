"""Shared base for entity-level naive predictors."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.naive._matrix_means import (
    additive_effects,
    predict_with_effects,
    require_pair_matrix,
    state_float_vector,
)
from drevalpy.components.predictors.structured import BlockPredictor
from drevalpy.components.state_helpers import state_float


class SingleEntityNaivePredictor(BlockPredictor):
    """Predict per-entity means from a one-hot design matrix."""

    requires_drug_featurizer: ClassVar[bool] = False
    _feature_side: ClassVar[str] = "cell_line"

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        super().__init__(hyperparameters)
        self._dataset_mean: float | None = None
        self._effects: np.ndarray | None = None

    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "Naive predictors require response values during fit"
            raise ValueError(msg)
        y = np.asarray(batch.response, dtype=np.float64)
        design = require_pair_matrix(batch, side=self._feature_side)
        self._dataset_mean = float(np.mean(y))
        self._effects = additive_effects(design, y, baseline=self._dataset_mean)

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        if self._dataset_mean is None or self._effects is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        design = require_pair_matrix(batch, side=self._feature_side)
        return predict_with_effects(design, self._effects, baseline=self._dataset_mean)

    def get_state(self) -> dict[str, object]:
        if self._dataset_mean is None or self._effects is None:
            return {}
        return {
            "dataset_mean": self._dataset_mean,
            "effects": self._effects.tolist(),
        }

    def set_state(self, state: dict[str, object]) -> None:
        mean = state_float(state, "dataset_mean")
        if mean is not None:
            self._dataset_mean = mean
        effects = state_float_vector(state, "effects")
        if effects is not None:
            self._effects = effects

    def is_fitted(self) -> bool:
        return self._dataset_mean is not None and self._effects is not None
