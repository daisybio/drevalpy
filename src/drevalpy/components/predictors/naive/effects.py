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
    description="Predict mean plus sample and drug effects.",
    sample_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class NaiveMeanEffectsPredictor(MeanEffectsStateMixin, BlockPredictor):
    """Naive mean effects predictor component."""

    required_sample_blocks: ClassVar[tuple[str, ...]] = ("identity",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("identity",)

    state_effects: ClassVar[tuple[str, ...]] = ("tissue_effects", "sample_effects", "drug_effects")

    _tissue_effects: np.ndarray | None
    _sample_effects: np.ndarray | None
    _drug_effects: np.ndarray | None

    def _cell_and_tissue(self, batch: ModelInputBatch) -> tuple[np.ndarray, np.ndarray]:
        """Return pair-aligned sample identity and tissue matrices; the tissue matrix has no columns if absent."""
        if "identity" in batch.sample.blocks:
            cell = block_pair_matrix(batch, "identity")
        else:
            cell = require_pair_matrix(batch, side="sample")
        if "tissue" in batch.sample.blocks:
            tissue = pair_align(batch.sample.blocks["tissue"].values, batch.sample_pair_idx)
        else:
            tissue = np.empty((batch.n_pairs, 0), dtype=np.float64)
        return np.asarray(cell, dtype=np.float64), np.asarray(tissue, dtype=np.float64)

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit on training data.

        :param batch: Training batch; tissue effects are fitted first, then sample effects on the residual, then drug
            effects.
        :raises RuntimeError: If ``batch.response`` is None.
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
            self._sample_effects = additive_effects(cell, residual, baseline=0.0)
        else:
            self._tissue_effects = np.empty((0,), dtype=np.float64)
            self._sample_effects = additive_effects(cell, y, baseline=self._dataset_mean)
        self._drug_effects = additive_effects(drugs, y, baseline=self._dataset_mean)

    def _predict(self, batch: ModelInputBatch) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param batch: Pairs to score.
        :returns: Dataset mean plus the tissue, sample and drug effects of each pair.
        :raises RuntimeError: If the predictor has not been fitted.
        """
        if (
            self._dataset_mean is None
            or self._tissue_effects is None
            or self._sample_effects is None
            or self._drug_effects is None
        ):
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        cell, tissue = self._cell_and_tissue(batch)
        drugs = np.asarray(require_pair_matrix(batch, side="drug"), dtype=np.float64)
        preds = np.full(batch.n_pairs, self._dataset_mean, dtype=np.float64)
        if cell.shape[1] > 0:
            preds = preds + cell @ self._sample_effects
        if tissue.shape[1] > 0 and self._tissue_effects.size > 0:
            preds = preds + tissue @ self._tissue_effects
        if drugs.shape[1] > 0:
            preds = preds + drugs @ self._drug_effects
        return preds
