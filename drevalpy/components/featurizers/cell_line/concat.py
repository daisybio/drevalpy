"""Concatenate outputs from multiple cell-line featurizers."""

from __future__ import annotations

from typing import Any

from drevalpy.components.featurizers._concat import ConcatFeaturizersMixin
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "concatFeaturizers",
    description="Concatenate dense outputs from multiple cell-line featurizers.",
    category="native",
)
class ConcatFeaturizersCellLineFeaturizer(ConcatFeaturizersMixin, CellLineFeaturizer):
    """Fit child featurizers independently and concatenate their dense outputs."""

    _not_fitted_msg = "ConcatFeaturizersCellLineFeaturizer must be fit before transform"

    def __init__(
        self,
        *,
        featurizers: list[Any] | None = None,
        registry: str = "cell_line",
    ) -> None:
        self._init_concat(featurizers=featurizers, registry=registry)

    @classmethod
    def distribute_legacy_state(
        cls,
        featurizer: ConcatFeaturizersCellLineFeaturizer,
        state: dict[str, object],
    ) -> None:
        """Map legacy flat preprocessing state onto child featurizers when possible."""
        from drevalpy.components.featurizers.cell_line.omics.methylation import MethylationPCACellLineFeaturizer
        from drevalpy.components.featurizers.cell_line.omics.proteomics import ProteomicsCellLineFeaturizer
        from drevalpy.components.featurizers.cell_line.omics.scaled_gene_expression import (
            ScaledGeneExpressionFeaturizer,
        )

        for name, child in featurizer._children:
            if isinstance(child, ScaledGeneExpressionFeaturizer):
                child_state = {
                    key: state[key]
                    for key in ("gene_expression_scaler", "fitted")
                    if key in state
                }
                if child_state:
                    child.set_state(child_state)
            elif isinstance(child, MethylationPCACellLineFeaturizer):
                child_state = {
                    key: state[key]
                    for key in ("methylation_scaler", "methylation_pca", "fitted")
                    if key in state
                }
                if child_state:
                    child.set_state(child_state)
            elif isinstance(child, ProteomicsCellLineFeaturizer):
                child_state = {
                    key: state[key]
                    for key in ("proteomics_transformer",)
                    if key in state
                }
                if child_state:
                    child.set_state(child_state)
            _ = name
        from drevalpy.models.featurizer_mapping import CELL_LINE_VIEW_TO_FEATURIZER

        if state.get("view_dims") and isinstance(state["view_dims"], dict):
            featurizer._block_dims = {
                CELL_LINE_VIEW_TO_FEATURIZER.get(str(key), str(key)): int(value)
                for key, value in state["view_dims"].items()
            }
        if state.get("output_dim"):
            featurizer._output_dim = int(state["output_dim"])
        if state.get("fitted"):
            featurizer._is_fitted = True
