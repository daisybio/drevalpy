"""Naive mean-effects predictor."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.feature_format import BlockSpec
from drevalpy.components.predictors.abstract.base import Predictor, get_response
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.naive._matrix_means import (
    additive_effects,
    block_pair_matrix,
    pair_align,
    require_pair_matrix,
)
from drevalpy.registry.predictor import register
from drevalpy.types.data.predictor_input import PredictorInput


@register(
    "naiveMeanEffects",
    tags=("baseline",),
    description="Predict mean plus sample and drug effects.",
)
class NaiveMeanEffectsPredictor(Predictor):
    """Naive mean effects predictor component."""

    sample_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("oneHot"))
    drug_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("oneHot"))

    _dataset_mean: float | None = None
    _tissue_effects: np.ndarray | None = None
    _sample_effects: np.ndarray | None = None
    _drug_effects: np.ndarray | None = None

    def _cell_and_tissue(self, data: PredictorInput) -> tuple[np.ndarray, np.ndarray]:
        """Return pair-aligned sample identity and tissue matrices; the tissue matrix has no columns if absent."""
        if "oneHot" in data.sample.blocks:
            cell = block_pair_matrix(data, "oneHot")
        else:
            cell = require_pair_matrix(data, side="sample")
        if "tissue" in data.sample.blocks:
            tissue = pair_align(data.sample.blocks["tissue"].values, data.sample_pair_idx)
        else:
            tissue = np.empty((len(data), 0), dtype=np.float64)
        return np.asarray(cell, dtype=np.float64), np.asarray(tissue, dtype=np.float64)

    def _fit(self, data: PredictorInput) -> None:
        """Fit on training data.

        :param data: Training batch; tissue effects are fitted first, then sample effects on the residual, then drug
            effects.
        """
        y = get_response(data)
        cell, tissue = self._cell_and_tissue(data)
        drugs = np.asarray(require_pair_matrix(data, side="drug"), dtype=np.float64)
        self._dataset_mean = float(np.mean(y))
        if tissue.shape[1] > 0:
            self._tissue_effects = additive_effects(tissue, y, baseline=self._dataset_mean)
            residual = y - self._dataset_mean - tissue @ self._tissue_effects
            self._sample_effects = additive_effects(cell, residual, baseline=0.0)
        else:
            self._tissue_effects = np.empty((0,), dtype=np.float64)
            self._sample_effects = additive_effects(cell, y, baseline=self._dataset_mean)
        self._drug_effects = additive_effects(drugs, y, baseline=self._dataset_mean)

    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param data: Pairs to score.
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
        cell, tissue = self._cell_and_tissue(data)
        drugs = np.asarray(require_pair_matrix(data, side="drug"), dtype=np.float64)
        preds = np.full(len(data), self._dataset_mean, dtype=np.float64)
        if cell.shape[1] > 0:
            preds = preds + cell @ self._sample_effects
        if tissue.shape[1] > 0 and self._tissue_effects.size > 0:
            preds = preds + tissue @ self._tissue_effects
        if drugs.shape[1] > 0:
            preds = preds + drugs @ self._drug_effects
        return preds
