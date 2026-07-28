"""SuperFELTR raw literature predictor registration."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.predictors.literature._metadata import SUPERFELTR_REFERENCE
from drevalpy.components.predictors.literature.raw_engine_adapter import RawLiteratureEnginePredictor
from drevalpy.components.registry import register_predictor
from drevalpy.models.config import ModelScope


@register_predictor(
    "superfeltr",
    description="SuperFELTR single-drug multi-omics model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=SUPERFELTR_REFERENCE,
)
class SuperFELTRPredictor(RawLiteratureEnginePredictor):
    """SuperFELTR predictor component."""

    supported_scopes: ClassVar[frozenset[ModelScope]] = frozenset({ModelScope.SINGLE_DRUG})
    required_cell_line_views: ClassVar[tuple[str, ...]] = (
        "gene_expression",
        "mutations",
        "copy_number_variation_gistic",
    )
    required_drug_views: ClassVar[tuple[str, ...]] = ()
    supports_early_stopping: ClassVar[bool] = True
    _engine_class_name = "SuperFELTR"
