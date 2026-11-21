"""Tissue-aware naive mean predictors."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.feature_format import BlockSpec
from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.naive._matrix_means import require_pair_matrix
from drevalpy.components.predictors.naive._single_entity import SingleEntityNaivePredictor
from drevalpy.registry.predictor import register
from drevalpy.types.data.predictor_input import PredictorInput


@register(
    "naiveTissueMean",
    tags=("baseline",),
    description="Predict per-tissue mean response with global fallback.",
)
class NaiveTissueMeanPredictor(SingleEntityNaivePredictor):
    """Naive tissue mean predictor component."""

    sample_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("tissue"))

    def _design(self, data: PredictorInput) -> np.ndarray:
        """Return the pair-level one-hot tissue matrix.

        :param data: Batch to read.
        :returns: One row per pair, one column per tissue.
        :raises ValueError: If the tissue featurizer produced no columns.
        """
        design = super()._design(data)
        if design.shape[1] == 0:
            msg = "NaiveTissueMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        return design


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

    def _fit(self, data: PredictorInput) -> None:
        """Fit on training data.

        :param data: Training batch.
        :raises ValueError: If the tissue or drug featurizer produced no columns.
        """
        tissue = require_pair_matrix(data, side="sample")
        drugs = require_pair_matrix(data, side="drug")
        if tissue.shape[1] == 0 or drugs.shape[1] == 0:
            msg = "NaiveTissueDrugMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        y = data.response
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

    def _predict(self, data: PredictorInput) -> np.ndarray:
        """Predict responses for each pair in the batch.

        :param data: Pairs to score.
        :returns: Dataset mean plus the effect of each pair's tissue-drug combination.
        :raises RuntimeError: If the predictor has not been fitted.
        :raises ValueError: If the tissue or drug featurizer produced no columns.
        """
        if self._dataset_mean is None or self._effects is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        tissue = require_pair_matrix(data, side="sample")
        drugs = require_pair_matrix(data, side="drug")
        if tissue.shape[1] == 0 or drugs.shape[1] == 0:
            msg = "NaiveTissueDrugMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        tissue64 = np.asarray(tissue, dtype=np.float64)
        drugs64 = np.asarray(drugs, dtype=np.float64)
        return self._dataset_mean + np.einsum("ni,ij,nj->n", tissue64, self._effects, drugs64)
