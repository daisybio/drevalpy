"""DIPK raw literature predictor registration."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.predictors.literature._metadata import DIPK_REFERENCE
from drevalpy.components.predictors.literature.raw_engine_adapter import RawLiteratureEnginePredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "dipk",
    description="DIPK BIONIC + MolGNet model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.RAGGED_SEQUENCE,
    reference=DIPK_REFERENCE,
)
class DIPKPredictor(RawLiteratureEnginePredictor):
    """DIPK predictor component."""

    required_cell_line_views: ClassVar[tuple[str, ...]] = ("gene_expression", "bionic_features")
    required_drug_views: ClassVar[tuple[str, ...]] = ("molgnet_features",)
    supports_early_stopping: ClassVar[bool] = True
    _engine_class_name = "DIPKModel"
