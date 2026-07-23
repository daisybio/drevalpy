"""PharmaFormer structured literature predictor registration."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.predictors.literature._metadata import PHARMAFORMER_METADATA
from drevalpy.components.predictors.literature.structured_engine_adapter import StructuredLiteratureEnginePredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "pharmaFormer",
    description="PharmaFormer landmark genes + BPE PharmaFormer model.",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
    **PHARMAFORMER_METADATA,
)
class PharmaFormerPredictor(StructuredLiteratureEnginePredictor):
    """Pharma former predictor component."""

    requires_raw_feature_datasets: ClassVar[bool] = True
    supports_early_stopping: ClassVar[bool] = True
    _engine_class_name = "PharmaFormerModel"
