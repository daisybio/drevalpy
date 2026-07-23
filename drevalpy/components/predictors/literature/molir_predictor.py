"""MOLIR structured literature predictor registration."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.predictors.literature._metadata import MOLIR_METADATA
from drevalpy.components.predictors.literature.structured_engine_adapter import StructuredLiteratureEnginePredictor
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import ModelScope


@register_predictor(
    "molir",
    description="MOLIR single-drug multi-omics model.",
    cell_line_contract=FeatureKind.DENSE,
    drug_contract=FeatureKind.DENSE,
    **MOLIR_METADATA,
)
class MOLIRPredictor(StructuredLiteratureEnginePredictor):
    """Molirpredictor component."""

    requires_drug_featurizer: ClassVar[bool] = False
    supported_scopes: ClassVar[frozenset[ModelScope]] = frozenset({ModelScope.SINGLE_DRUG})
    requires_raw_feature_datasets: ClassVar[bool] = True
    supports_early_stopping: ClassVar[bool] = True
    _engine_class_name = "MOLIR"
