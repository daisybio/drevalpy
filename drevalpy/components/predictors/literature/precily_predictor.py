"""Precily structured literature predictor registration."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.predictors.literature._metadata import PRECILY_REFERENCE
from drevalpy.components.predictors.literature.block_engine_adapter import BlockLiteratureEnginePredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "precily",
    description="Precily pathway + SMILESVec model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=PRECILY_REFERENCE,
)
class PrecilyPredictor(BlockLiteratureEnginePredictor):
    """Precily predictor component."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("pathways",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("smilesvec",)
    _engine_class_name = "PrecilyModel"
