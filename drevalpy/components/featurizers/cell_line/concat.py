"""Concatenate outputs from multiple cell-line featurizers."""

from __future__ import annotations

from typing import Any

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.featurizers._concat import ConcatFeaturizersMixin
from drevalpy.components.featurizers.cell_line.base import CellLineFeaturizer
from drevalpy.components.registry import register_cell_line_featurizer


@register_cell_line_featurizer(
    "concatFeaturizers",
    description="Concatenate dense outputs from multiple cell-line featurizers.",
    category="native",
    contract=FeatureKind.DENSE,
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

        if not featurizer._children:
            featurizer._materialize_children()

        for name, child in featurizer._children:
            if isinstance(child, ScaledGeneExpressionFeaturizer):
                child_state = {key: state[key] for key in ("gene_expression_scaler", "fitted") if key in state}
                if child_state:
                    child.set_state(child_state)
            elif isinstance(child, MethylationPCACellLineFeaturizer):
                child_state = {
                    key: state[key] for key in ("methylation_scaler", "methylation_pca", "fitted") if key in state
                }
                if child_state:
                    child.set_state(child_state)
            elif isinstance(child, ProteomicsCellLineFeaturizer):
                child_state = {key: state[key] for key in ("proteomics_transformer",) if key in state}
                if child_state:
                    child.set_state(child_state)
            _ = name
        from drevalpy.models.featurizer_mapping import CELL_LINE_VIEW_TO_FEATURIZER

        if state.get("view_dims") and isinstance(state["view_dims"], dict):
            featurizer._block_dims = {
                CELL_LINE_VIEW_TO_FEATURIZER.get(str(key), str(key)): int(value)
                for key, value in state["view_dims"].items()
            }
        output_dim = state.get("output_dim")
        if isinstance(output_dim, int):
            featurizer._output_dim = output_dim
        if state.get("fitted"):
            featurizer._is_fitted = True

    @classmethod
    def collect_legacy_state(
        cls,
        featurizer: ConcatFeaturizersCellLineFeaturizer,
    ) -> dict[str, object]:
        """Flatten child featurizer state for legacy sklearn save/load."""
        if not featurizer._children:
            featurizer._materialize_children()
        state: dict[str, object] = {"fitted": featurizer._is_fitted}
        for _name, child in featurizer._children:
            child_state = child.get_state()
            if not isinstance(child_state, dict):
                continue
            for key, value in child_state.items():
                if key != "fitted":
                    state[key] = value
        if featurizer._block_dims:
            from drevalpy.models.featurizer_mapping import CELL_LINE_VIEW_TO_FEATURIZER

            featurizer_to_view = {value: key for key, value in CELL_LINE_VIEW_TO_FEATURIZER.items()}
            view_dims = {featurizer_to_view.get(name, name): dim for name, dim in featurizer._block_dims.items()}
            state["view_dims"] = view_dims
        if featurizer._output_dim:
            state["output_dim"] = featurizer._output_dim
        return state
