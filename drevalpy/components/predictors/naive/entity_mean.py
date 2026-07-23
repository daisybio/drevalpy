"""Per-entity naive mean predictors."""

from __future__ import annotations

import numpy as np

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.naive._single_entity import SingleEntityNaivePredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "naiveDrugMean",
    description="Predict per-drug mean response with global fallback.",
    category="baseline",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
)
class NaiveDrugMeanPredictor(SingleEntityNaivePredictor):
    """Naive drug mean predictor component."""

    def _entity_keys(self, batch: ModelInputBatch) -> np.ndarray:
        return batch.drug_ids

    def _legacy_entity_means_key(self) -> str:
        return "drug_means"


@register_predictor(
    "naiveCellLineMean",
    description="Predict per-cell-line mean response with global fallback.",
    category="baseline",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
)
class NaiveCellLineMeanPredictor(SingleEntityNaivePredictor):
    """Naive cell line mean predictor component."""

    def _entity_keys(self, batch: ModelInputBatch) -> np.ndarray:
        return batch.cell_line_ids

    def _legacy_entity_means_key(self) -> str:
        return "cell_line_means"
