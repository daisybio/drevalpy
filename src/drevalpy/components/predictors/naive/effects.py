"""Naive mean-effects predictor."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.naive._category_means import (
    additive_effects,
    block_pair_codes,
    effects_of,
    require_pair_codes,
)
from drevalpy.registry.predictor import register
from drevalpy.types.data.feature_format import BlockSpec
from drevalpy.types.data.feature_source import Side
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

    def _cell_and_tissue(self, data: PredictorInput) -> tuple[tuple[np.ndarray, int], tuple[np.ndarray, int]]:
        """Return the category codes and category counts of the sample and tissue of each pair.

        The tissue has no categories if the batch has no tissue block.
        """
        if "oneHot" in data.sample.blocks:
            cell = block_pair_codes(data, "oneHot")
        else:
            cell = require_pair_codes(data, side=Side.SAMPLE)
        if "tissue" in data.sample.blocks:
            tissue = block_pair_codes(data, "tissue")
        else:
            tissue = (np.full(len(data), -1, dtype=np.int64), 0)
        return cell, tissue

    def _fit(self, data: PredictorInput) -> None:
        """Fit on training data.

        :param data: Training batch; tissue effects are fitted first, then sample effects on the residual, then drug
            effects.
        """
        y = data.response
        (cell, n_cells), (tissue, n_tissues) = self._cell_and_tissue(data)
        drugs, n_drugs = require_pair_codes(data, side=Side.DRUG)
        self._dataset_mean = float(np.mean(y))
        if n_tissues > 0:
            self._tissue_effects = additive_effects(tissue, n_tissues, y, baseline=self._dataset_mean)
            residual = y - self._dataset_mean - effects_of(tissue, self._tissue_effects)
            self._sample_effects = additive_effects(cell, n_cells, residual, baseline=0.0)
        else:
            self._tissue_effects = np.empty((0,), dtype=np.float64)
            self._sample_effects = additive_effects(cell, n_cells, y, baseline=self._dataset_mean)
        self._drug_effects = additive_effects(drugs, n_drugs, y, baseline=self._dataset_mean)

    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param data: Pairs to score.
        :returns: Dataset mean plus the tissue, sample and drug effects of each pair; an entity without a category
            (an all-zero row) adds no effect.
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
        (cell, _), (tissue, _) = self._cell_and_tissue(data)
        drugs, _ = require_pair_codes(data, side=Side.DRUG)
        return (
            self._dataset_mean
            + effects_of(cell, self._sample_effects)
            + effects_of(tissue, self._tissue_effects)
            + effects_of(drugs, self._drug_effects)
        )
