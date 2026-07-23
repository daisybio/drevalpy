"""SRMF structured literature predictor registration."""

from __future__ import annotations

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.predictors.literature._metadata import SRMF_METADATA
from drevalpy.components.predictors.literature.structured_engine_adapter import StructuredLiteratureEnginePredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "srmf",
    description="SRMF matrix factorization model.",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
    **SRMF_METADATA,
)
class SRMFPredictor(StructuredLiteratureEnginePredictor):
    """Srmfpredictor component."""

    _engine_class_name = "SRMF"
