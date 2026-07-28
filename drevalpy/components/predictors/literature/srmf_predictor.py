"""SRMF structured literature predictor registration."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.predictors.literature._metadata import SRMF_REFERENCE
from drevalpy.components.predictors.literature.block_engine_adapter import BlockLiteratureEnginePredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "srmf",
    description="SRMF matrix factorization model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=SRMF_REFERENCE,
)
class SRMFPredictor(BlockLiteratureEnginePredictor):
    """SRMF predictor component."""

    required_cell_line_blocks: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_blocks: ClassVar[tuple[str, ...]] = ("fingerprints",)
    _engine_class_name = "SRMF"
