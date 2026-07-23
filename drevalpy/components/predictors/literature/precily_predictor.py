"""Precily structured literature predictor registration."""

from __future__ import annotations

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.predictors.literature._metadata import PRECIILY_METADATA
from drevalpy.components.predictors.literature.structured_engine_adapter import StructuredLiteratureEnginePredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "precily",
    description="Precily pathway + SMILESVec model.",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
    **PRECIILY_METADATA,
)
class PrecilyPredictor(StructuredLiteratureEnginePredictor):
    """Precily predictor component."""

    _engine_class_name = "PrecilyModel"
