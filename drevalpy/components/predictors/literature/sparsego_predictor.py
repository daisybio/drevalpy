"""SparseGO structured literature predictor registration."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.predictors.literature._metadata import SPARSEGO_METADATA
from drevalpy.components.predictors.literature.structured_engine_adapter import StructuredLiteratureEnginePredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "sparsego",
    description="SparseGO GO-structured visible neural network.",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
    **SPARSEGO_METADATA,
)
class SparseGOPredictor(StructuredLiteratureEnginePredictor):
    """SparseGO predictor component."""

    requires_raw_feature_datasets: ClassVar[bool] = True
    _engine_class_name = "SparseGOModel"
