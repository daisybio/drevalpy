"""Naive mean-effects predictor."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors._state_helpers import state_float
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.naive._matrix_means import (
    additive_effects,
    block_pair_matrix,
    pair_align,
    require_pair_matrix,
    state_float_vector,
)
from drevalpy.components.registry import register_predictor
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


@register_predictor(
    "naiveMeanEffects",
    tags=("baseline",),
    description="Predict mean plus cell-line and drug effects.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class NaiveMeanEffectsPredictor(BlockPredictor):
    """Naive mean effects predictor component."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("identity",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("identity",)

    def __init__(self, hyperparameters: dict[str, Any] | None = None) -> None:
        """Initialize instance state.

        :param hyperparameters: hyperparameters.
        """
        super().__init__(hyperparameters)
        self._dataset_mean: float | None = None
        self._tissue_effects: np.ndarray | None = None
        self._cell_line_effects: np.ndarray | None = None
        self._drug_effects: np.ndarray | None = None

    def _cell_and_tissue(self, batch: ModelInputBatch) -> tuple[np.ndarray, np.ndarray]:
        if "identity" in batch.cell_line_blocks:
            cell = block_pair_matrix(batch, "identity")
        else:
            cell = require_pair_matrix(batch, side="cell_line")
        if "tissue" in batch.cell_line_blocks:
            tissue = pair_align(batch.cell_line_blocks["tissue"].values, batch.cell_line_pair_idx)
        else:
            tissue = np.empty((batch.n_pairs, 0), dtype=np.float64)
        return np.asarray(cell, dtype=np.float64), np.asarray(tissue, dtype=np.float64)

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit on training data.

        :param batch: batch.
        :raises RuntimeError: If batch.response is None.
        """
        y = batch.response
        if y is None:
            raise RuntimeError("batch.response is required for fit")
        cell, tissue = self._cell_and_tissue(batch)
        drugs = np.asarray(require_pair_matrix(batch, side="drug"), dtype=np.float64)
        self._dataset_mean = float(np.mean(y))
        if tissue.shape[1] > 0:
            self._tissue_effects = additive_effects(tissue, y, baseline=self._dataset_mean)
            residual = y - self._dataset_mean - tissue @ self._tissue_effects
            self._cell_line_effects = additive_effects(cell, residual, baseline=0.0)
        else:
            self._tissue_effects = np.empty((0,), dtype=np.float64)
            self._cell_line_effects = additive_effects(cell, y, baseline=self._dataset_mean)
        self._drug_effects = additive_effects(drugs, y, baseline=self._dataset_mean)

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param batch: batch.
        :returns: Result.
        :raises RuntimeError: Raised on invalid input.
        """
        if (
            self._dataset_mean is None
            or self._tissue_effects is None
            or self._cell_line_effects is None
            or self._drug_effects is None
        ):
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        cell, tissue = self._cell_and_tissue(batch)
        drugs = np.asarray(require_pair_matrix(batch, side="drug"), dtype=np.float64)
        preds = np.full(batch.n_pairs, self._dataset_mean, dtype=np.float64)
        if cell.shape[1] > 0:
            preds = preds + cell @ self._cell_line_effects
        if tissue.shape[1] > 0 and self._tissue_effects.size > 0:
            preds = preds + tissue @ self._tissue_effects
        if drugs.shape[1] > 0:
            preds = preds + drugs @ self._drug_effects
        return preds

    def get_state(self) -> dict[str, object]:
        """Return serializable fitted state.

        :returns: Result.
        """
        if (
            self._dataset_mean is None
            or self._tissue_effects is None
            or self._cell_line_effects is None
            or self._drug_effects is None
        ):
            return {}
        return {
            "dataset_mean": self._dataset_mean,
            "tissue_effects": self._tissue_effects.tolist(),
            "cell_line_effects": self._cell_line_effects.tolist(),
            "drug_effects": self._drug_effects.tolist(),
        }

    def set_state(self, state: dict[str, object]) -> None:
        """Restore state from a prior ``get_state`` mapping.

        :param state: state.
        """
        mean = state_float(state, "dataset_mean")
        if mean is not None:
            self._dataset_mean = mean
        tissue_effects = state_float_vector(state, "tissue_effects")
        if tissue_effects is not None:
            self._tissue_effects = tissue_effects
        cell_line_effects = state_float_vector(state, "cell_line_effects")
        if cell_line_effects is not None:
            self._cell_line_effects = cell_line_effects
        drug_effects = state_float_vector(state, "drug_effects")
        if drug_effects is not None:
            self._drug_effects = drug_effects

    def is_fitted(self) -> bool:
        """Return whether the component has been fit.

        :returns: Result.
        """
        return (
            self._dataset_mean is not None
            and self._tissue_effects is not None
            and self._cell_line_effects is not None
            and self._drug_effects is not None
        )
