"""Per-entity naive mean predictors."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureKind
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

    requires_drug_featurizer: ClassVar[bool] = True
    _feature_side: ClassVar[str] = "drug"


@register_predictor(
    "naiveCellLineMean",
    description="Predict per-cell-line mean response with global fallback.",
    category="baseline",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
)
class NaiveCellLineMeanPredictor(SingleEntityNaivePredictor):
    """Naive cell line mean predictor component."""

    _feature_side: ClassVar[str] = "cell_line"
