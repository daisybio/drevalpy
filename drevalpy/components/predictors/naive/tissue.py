"""Tissue-aware naive mean predictors."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.core.batch.model_input_batch import ModelInputBatch
from drevalpy.components.predictors._state_helpers import state_float
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.naive._matrix_means import (
    additive_effects,
    predict_with_effects,
    require_pair_matrix,
    state_float_matrix,
    state_float_vector,
)
from drevalpy.components.registry import register_predictor


@register_predictor(
    "naiveTissueMean",
    tags=("baseline",),
    description="Predict per-tissue mean response with global fallback.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class NaiveTissueMeanPredictor(BlockPredictor):
    """Naive tissue mean predictor component."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("tissue",)

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize instance state.

        :param hyperparameters: hyperparameters.
        """
        super().__init__(hyperparameters)
        self._dataset_mean: float | None = None
        self._effects: np.ndarray | None = None

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit on training data.

        :param batch: batch.
        :raises ValueError: Raised on invalid input.
        :raises RuntimeError: If batch.response is None.
        """
        design = require_pair_matrix(batch, side="cell_line")
        if design.shape[1] == 0:
            msg = "NaiveTissueMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        y = batch.response
        if y is None:
            raise RuntimeError("batch.response is required for fit")
        self._dataset_mean = float(np.mean(y))
        self._effects = additive_effects(design, y, baseline=self._dataset_mean)

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param batch: batch.
        :returns: Result.
        :raises RuntimeError: Raised on invalid input.
        :raises ValueError: Raised on invalid input.
        """
        if self._dataset_mean is None or self._effects is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        design = require_pair_matrix(batch, side="cell_line")
        if design.shape[1] == 0:
            msg = "NaiveTissueMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        return predict_with_effects(design, self._effects, baseline=self._dataset_mean)

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        if self._dataset_mean is None or self._effects is None:
            return {}
        return {
            "dataset_mean": self._dataset_mean,
            "effects": self._effects.tolist(),
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        mean = state_float(state, "dataset_mean")
        if mean is not None:
            self._dataset_mean = mean
        effects = state_float_vector(state, "effects")
        if effects is not None:
            self._effects = effects

    def is_fitted(self) -> bool:
        """Return whether the component has been fit.

        :returns: Result.
        """
        return self._dataset_mean is not None and self._effects is not None


@register_predictor(
    "naiveTissueDrugMean",
    tags=("baseline",),
    description="Predict per tissue-drug combination mean response.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class NaiveTissueDrugMeanPredictor(BlockPredictor):
    """Naive tissue drug mean predictor component."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("tissue",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("identity",)

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize instance state.

        :param hyperparameters: hyperparameters.
        """
        super().__init__(hyperparameters)
        self._dataset_mean: float | None = None
        self._effects: np.ndarray | None = None

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit on training data.

        :param batch: batch.
        :raises ValueError: Raised on invalid input.
        :raises RuntimeError: If batch.response is None.
        """
        tissue = require_pair_matrix(batch, side="cell_line")
        drugs = require_pair_matrix(batch, side="drug")
        if tissue.shape[1] == 0 or drugs.shape[1] == 0:
            msg = "NaiveTissueDrugMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        y = batch.response
        if y is None:
            raise RuntimeError("batch.response is required for fit")
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

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param batch: batch.
        :returns: Result.
        :raises RuntimeError: Raised on invalid input.
        :raises ValueError: Raised on invalid input.
        """
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
        """Return serializable fitted state.

        :returns: Result.
        """
        if self._dataset_mean is None or self._effects is None:
            return {}
        return {
            "dataset_mean": self._dataset_mean,
            "effects": self._effects.tolist(),
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        mean = state_float(state, "dataset_mean")
        if mean is not None:
            self._dataset_mean = mean
        effects = state_float_matrix(state, "effects")
        if effects is not None:
            self._effects = effects

    def is_fitted(self) -> bool:
        """Return whether the component has been fit.

        :returns: Result.
        """
        return self._dataset_mean is not None and self._effects is not None
