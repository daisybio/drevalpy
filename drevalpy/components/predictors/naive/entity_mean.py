"""Per-entity naive mean predictors."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.core.contracts.contracts import FeatureFormat
from drevalpy.components.predictors.naive._single_entity import SingleEntityNaivePredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "naiveDrugMean",
    tags=("baseline",),
    description="Predict per-drug mean response with global fallback.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class NaiveDrugMeanPredictor(SingleEntityNaivePredictor):
    """Naive drug mean predictor component."""

    _feature_side: ClassVar[str] = "drug"


@register_predictor(
    "naiveCellLineMean",
    tags=("baseline",),
    description="Predict per-cell-line mean response with global fallback.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
)
class NaiveCellLineMeanPredictor(SingleEntityNaivePredictor):
    """Naive cell line mean predictor component."""

    _feature_side: ClassVar[str] = "cell_line"
