"""Tissue-aware naive mean predictors."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.naive._matrix_means import (
    additive_effects,
    predict_with_effects,
    require_pair_matrix,
    state_float_matrix,
    state_float_vector,
)
from drevalpy.components.predictors.structured import BlockPredictor
from drevalpy.components.registry import register_predictor
from drevalpy.components.state_helpers import state_float


@register_predictor(
    "naiveTissueMean",
    description="Predict per-tissue mean response with global fallback.",
    category="baseline",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
)
class NaiveTissueMeanPredictor(BlockPredictor):
    """Naive tissue mean predictor component."""

    requires_drug_featurizer: ClassVar[bool] = False

    def __init__(self) -> None:
        self._dataset_mean: float | None = None
        self._effects: np.ndarray | None = None

    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "Naive predictors require response values during fit"
            raise ValueError(msg)
        design = require_pair_matrix(batch, side="cell_line")
        if design.shape[1] == 0:
            msg = "NaiveTissueMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        y = np.asarray(batch.response, dtype=np.float64)
        self._dataset_mean = float(np.mean(y))
        self._effects = additive_effects(design, y, baseline=self._dataset_mean)

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        if self._dataset_mean is None or self._effects is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        design = require_pair_matrix(batch, side="cell_line")
        if design.shape[1] == 0:
            msg = "NaiveTissueMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
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


@register_predictor(
    "naiveTissueDrugMean",
    description="Predict per tissue-drug combination mean response.",
    category="baseline",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
)
class NaiveTissueDrugMeanPredictor(BlockPredictor):
    """Naive tissue drug mean predictor component."""

    requires_drug_featurizer: ClassVar[bool] = True

    def __init__(self) -> None:
        self._dataset_mean: float | None = None
        self._effects: np.ndarray | None = None

    def fit(self, batch: ModelInputBatch) -> None:
        if batch.response is None:
            msg = "Naive predictors require response values during fit"
            raise ValueError(msg)
        tissue = require_pair_matrix(batch, side="cell_line")
        drugs = require_pair_matrix(batch, side="drug")
        if tissue.shape[1] == 0 or drugs.shape[1] == 0:
            msg = "NaiveTissueDrugMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        y = np.asarray(batch.response, dtype=np.float64)
        self._dataset_mean = float(np.mean(y))
        tissue64 = np.asarray(tissue, dtype=np.float64)
        drugs64 = np.asarray(drugs, dtype=np.float64)
        counts = tissue64.T @ drugs64
        sums = tissue64.T @ (drugs64 * y[:, None])
        effects = np.zeros_like(counts, dtype=np.float64)
        np.divide(sums, counts, out=effects, where=counts > 0)
        effects = effects - self._dataset_mean
        effects = np.where(counts > 0, effects, 0.0)
        self._effects = effects

    def predict(self, batch: ModelInputBatch) -> np.ndarray:
        if self._dataset_mean is None or self._effects is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        tissue = require_pair_matrix(batch, side="cell_line")
        drugs = require_pair_matrix(batch, side="drug")
        if tissue.shape[1] == 0 or drugs.shape[1] == 0:
            msg = "NaiveTissueDrugMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        tissue64 = np.asarray(tissue, dtype=np.float64)
        drugs64 = np.asarray(drugs, dtype=np.float64)
        return self._dataset_mean + np.einsum("ni,ij,nj->n", tissue64, self._effects, drugs64)

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
        effects = state_float_matrix(state, "effects")
        if effects is not None:
            self._effects = effects

    def is_fitted(self) -> bool:
        return self._dataset_mean is not None and self._effects is not None
