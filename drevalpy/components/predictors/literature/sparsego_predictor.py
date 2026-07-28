"""SparseGO raw literature predictor registration."""

from __future__ import annotations

from typing import ClassVar

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.predictors.literature._metadata import SPARSEGO_REFERENCE
from drevalpy.components.predictors.literature.raw_engine_adapter import RawLiteratureEnginePredictor
from drevalpy.components.registry import register_predictor


@register_predictor(
    "sparsego",
    description="SparseGO GO-structured visible neural network.",
    cell_line_contract=FeatureFormat.NUMERIC_MATRIX,
    drug_contract=FeatureFormat.NUMERIC_MATRIX,
    reference=SPARSEGO_REFERENCE,
)
class SparseGOPredictor(RawLiteratureEnginePredictor):
    """SparseGO predictor component."""

    # Discovery lists both selectable omics modes; runtime uses ``input_type``.
    required_cell_line_views: ClassVar[tuple[str, ...]] = ("gene_expression", "mutations")
    required_drug_views: ClassVar[tuple[str, ...]] = ("fingerprints",)
    _engine_class_name = "SparseGOModel"

    def active_cell_line_views(self) -> tuple[str, ...]:
        input_type = str(self._hyperparameters.get("input_type", "expression"))
        if input_type == "mutations":
            return ("mutations",)
        return ("gene_expression",)
