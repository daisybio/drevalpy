"""Tissue-aware naive mean predictors."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.predictors.abstract.base import Predictor, fit_response
from drevalpy.components.predictors.inputs import Inputs
from drevalpy.components.predictors.naive._matrix_means import require_pair_matrix
from drevalpy.components.predictors.naive._single_entity import SingleEntityNaivePredictor
from drevalpy.registry.predictor import register
from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


@register(
    "naiveTissueMean",
    tags=("baseline",),
    description="Predict per-tissue mean response with global fallback.",
)
class NaiveTissueMeanPredictor(SingleEntityNaivePredictor):
    """Naive tissue mean predictor component."""

    sample_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("tissue", FeatureFormat.NUMERIC_MATRIX))

    def _design(self, batch: ModelInputBatch) -> np.ndarray:
        """Return the pair-level one-hot tissue matrix.

        :param batch: Batch to read.
        :returns: One row per pair, one column per tissue.
        :raises ValueError: If the tissue featurizer produced no columns.
        """
        design = super()._design(batch)
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

    sample_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("tissue", FeatureFormat.NUMERIC_MATRIX))
    drug_inputs: ClassVar[Inputs] = Inputs.blocks(BlockSpec("oneHot", FeatureFormat.NUMERIC_MATRIX))

    _dataset_mean: float | None = None
    _effects: np.ndarray | None = None

    def _fit(self, batch: ModelInputBatch) -> None:
        """Fit on training data.

        :param batch: Training batch.
        :raises ValueError: If the tissue or drug featurizer produced no columns.
        """
        tissue = require_pair_matrix(batch, side="sample")
        drugs = require_pair_matrix(batch, side="drug")
        if tissue.shape[1] == 0 or drugs.shape[1] == 0:
            msg = "NaiveTissueDrugMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        y = fit_response(batch)
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

        :param batch: Pairs to score.
        :returns: Dataset mean plus the effect of each pair's tissue-drug combination.
        :raises RuntimeError: If the predictor has not been fitted.
        :raises ValueError: If the tissue or drug featurizer produced no columns.
        """
        if self._dataset_mean is None or self._effects is None:
            msg = "Call fit before predict"
            raise RuntimeError(msg)
        tissue = require_pair_matrix(batch, side="sample")
        drugs = require_pair_matrix(batch, side="drug")
        if tissue.shape[1] == 0 or drugs.shape[1] == 0:
            msg = "NaiveTissueDrugMeanPredictor requires tissue featurizer output"
            raise ValueError(msg)
        tissue64 = np.asarray(tissue, dtype=np.float64)
        drugs64 = np.asarray(drugs, dtype=np.float64)
        return self._dataset_mean + np.einsum("ni,ij,nj->n", tissue64, self._effects, drugs64)
