"""Tissue-aware naive mean predictors."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.naive._category_means import (
    additive_effects,
    effects_of,
    joint_codes,
    require_pair_codes,
)
from drevalpy.components.predictors.naive._single_entity import SingleEntityNaivePredictor
from drevalpy.registry.predictor import register
from drevalpy.types.data.feature_format import BlockSpec
from drevalpy.types.data.feature_source import Side
from drevalpy.types.data.predictor_input import PredictorInput


@register(
    "naiveTissueMean",
    tags=("baseline",),
    description="Predict per-tissue mean response with global fallback.",
)
class NaiveTissueMeanPredictor(SingleEntityNaivePredictor):
    """Naive tissue mean predictor component."""

    sample_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("tissue"))

    def _codes(self, data: PredictorInput) -> tuple[np.ndarray, int]:
        """Return the tissue code of each pair.

        :param data: Batch to read.
        :returns: One code per pair (``-1`` for a sample without a tissue) and the number of tissues.
        :raises ValueError: If the tissue featurizer produced no columns.
        """
        codes, n_tissues = super()._codes(data)
        if n_tissues == 0:
            msg = "NaiveTissueMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        return codes, n_tissues


@register(
    "naiveTissueDrugMean",
    tags=("baseline",),
    description="Predict per tissue-drug combination mean response.",
)
class NaiveTissueDrugMeanPredictor(Predictor):
    """Naive tissue drug mean predictor component."""

    sample_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("tissue"))
    drug_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("oneHot"))

    _dataset_mean: float | None = None
    _effects: np.ndarray | None = None

    def _combination_codes(self, data: PredictorInput) -> tuple[np.ndarray, int]:
        """Return the tissue-drug combination code of each pair.

        :param data: Batch to read.
        :returns: One code per pair (``-1`` if its tissue or drug has no category) and the number of combinations.
        :raises ValueError: If the tissue or drug featurizer produced no columns.
        """
        tissue, n_tissues = require_pair_codes(data, side=Side.SAMPLE)
        drugs, n_drugs = require_pair_codes(data, side=Side.DRUG)
        if n_tissues == 0 or n_drugs == 0:
            msg = "NaiveTissueDrugMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        return joint_codes(tissue, n_tissues, drugs, n_drugs)

    def _fit(self, data: PredictorInput) -> None:
        """Fit on training data.

        :param data: Training batch.
        """
        codes, n_combinations = self._combination_codes(data)
        self._dataset_mean = float(np.mean(data.response))
        self._effects = additive_effects(codes, n_combinations, data.response, baseline=self._dataset_mean)

    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param data: Pairs to score.
        :returns: Dataset mean plus the effect of each pair's tissue-drug combination; the dataset mean for a
            combination without training pairs.
        :raises RuntimeError: If the predictor has not been fitted.
        """
        if self._dataset_mean is None or self._effects is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        codes, _ = self._combination_codes(data)
        return self._dataset_mean + effects_of(codes, self._effects)
