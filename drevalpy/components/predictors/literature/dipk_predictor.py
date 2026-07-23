"""DIPK structured literature predictor registration."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.predictors.literature._metadata import DIPK_METADATA
from drevalpy.components.predictors.literature.structured_engine_adapter import StructuredLiteratureEnginePredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "dipk",
    description="DIPK BIONIC + MolGNet model.",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
    **DIPK_METADATA,
)
class DIPKPredictor(StructuredLiteratureEnginePredictor):
    """Dipkpredictor component."""

    requires_raw_feature_datasets: ClassVar[bool] = True
    supports_early_stopping: ClassVar[bool] = True
    _engine_class_name = "DIPKModel"
