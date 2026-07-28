"""PharmaFormer raw literature predictor registration."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.predictors.literature._metadata import PHARMAFORMER_REFERENCE
from drevalpy.components.predictors.literature.raw_engine_adapter import RawLiteratureEnginePredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "pharmaFormer",
    description="PharmaFormer landmark genes + BPE PharmaFormer model.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=PHARMAFORMER_REFERENCE,
)
class PharmaFormerPredictor(RawLiteratureEnginePredictor):
    """PharmaFormer predictor component."""

    required_cell_line_views: ClassVar[tuple[str, ...]] = ("gene_expression",)
    required_drug_views: ClassVar[tuple[str, ...]] = ("bpe_smiles",)
    supports_early_stopping: ClassVar[bool] = True
    _engine_class_name = "PharmaFormerModel"
