"""Per-entity naive mean predictors."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.predictors.naive._single_entity import SingleEntityNaivePredictor
from drevalpy.registry.predictor import register


@register(
    "naiveDrugMean",
    tags=("baseline",),
    description="Predict per-drug mean response with global fallback.",
)
class NaiveDrugMeanPredictor(SingleEntityNaivePredictor):
    """Naive drug mean predictor component."""

    _feature_side: ClassVar[str] = "drug"


@register(
    "naiveSampleMean",
    tags=("baseline",),
    description="Predict per-sample mean response with global fallback.",
)
class NaiveSampleMeanPredictor(SingleEntityNaivePredictor):
    """Naive sample mean predictor component."""

    _feature_side: ClassVar[str] = "sample"
