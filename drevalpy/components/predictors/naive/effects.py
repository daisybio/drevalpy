"""Naive mean-effects predictor."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.naive._matrix_means import (
    additive_effects,
    block_pair_matrix,
    pair_align,
    require_pair_matrix,
)
from drevalpy.components.predictors.naive._state_mixin import MeanEffectsStateMixin
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


@register(
    "naiveMeanEffects",
    tags=("baseline",),
    description="Predict mean plus cell-line and drug effects.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class NaiveMeanEffectsPredictor(MeanEffectsStateMixin, BlockPredictor):
    """Naive mean effects predictor component."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("identity",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("identity",)

    state_effects: ClassVar[tuple[str, ...]] = ("tissue_effects", "cell_line_effects", "drug_effects")

    _tissue_effects: np.ndarray | None
    _cell_line_effects: np.ndarray | None
    _drug_effects: np.ndarray | None

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
