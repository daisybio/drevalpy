"""Translate concat-featurizer state to and from legacy flat attributes."""

from __future__ import annotations

from drevalpy.components.featurizers.cell_line.concat import ConcatFeaturizersCellLineFeaturizer
from drevalpy.components.featurizers.cell_line.normalized_proteomics import (
    NormalizedProteomicsCellLineFeaturizer,
)
from drevalpy.components.featurizers.cell_line.pca import PCACellLineFeaturizer
from drevalpy.components.featurizers.cell_line.scaled_gene_expression import (
    ScaledGeneExpressionFeaturizer,
)
from drevalpy.models.featurizer_mapping import CELL_LINE_VIEW_TO_FEATURIZER, view_to_concat_block_label


def restore_legacy_concat_state(
    featurizer: ConcatFeaturizersCellLineFeaturizer,
    state: dict[str, object],
) -> None:
    """Map legacy flat preprocessing state onto concat child featurizers."""
    if not featurizer._children:
        featurizer._materialize_children()

    for _name, child in featurizer._children:
        if isinstance(child, ScaledGeneExpressionFeaturizer):
            scaled_state = {key: state[key] for key in ("gene_expression_scaler", "fitted") if key in state}
            if scaled_state:
                child.set_state(scaled_state)
        elif isinstance(child, PCACellLineFeaturizer) and "methylation_pca" in state:
            methylation_pca = state["methylation_pca"]
            pca_state: dict[str, object] = {
                "pca": methylation_pca,
                "view": child._view,
                "fitted": state.get("fitted"),
            }
            if hasattr(methylation_pca, "n_components"):
                pca_state["n_components"] = int(methylation_pca.n_components)
                pca_state["output_dim"] = int(methylation_pca.n_components)
            child.set_state(pca_state)
        elif isinstance(child, NormalizedProteomicsCellLineFeaturizer):
            proteomics_state = {key: state[key] for key in ("proteomics_transformer",) if key in state}
            if proteomics_state:
                child.set_state(proteomics_state)

    view_dims = state.get("view_dims")
    if isinstance(view_dims, dict) and view_dims:
        featurizer._block_dims = {
            view_to_concat_block_label(str(key)): int(value) for key, value in view_dims.items()
        }
    output_dim = state.get("output_dim")
    if isinstance(output_dim, int):
        featurizer._output_dim = output_dim
    if state.get("fitted"):
        featurizer._is_fitted = True


def collect_legacy_concat_state(
    featurizer: ConcatFeaturizersCellLineFeaturizer,
) -> dict[str, object]:
    """Flatten concat child state for legacy wrapper attributes."""
    if not featurizer._children:
        featurizer._materialize_children()

    state: dict[str, object] = {"fitted": featurizer._is_fitted}
    for _name, child in featurizer._children:
        child_state = child.get_state()
        if not isinstance(child_state, dict):
            continue
        if isinstance(child, PCACellLineFeaturizer) and child._view == "methylation":
            pca = child_state.get("pca")
            if pca is not None:
                state["methylation_pca"] = pca
        for key, value in child_state.items():
            if key != "fitted":
                state.setdefault(key, value)

    if featurizer._block_dims:
        block_label_to_view = {
            view_to_concat_block_label(view): view for view in CELL_LINE_VIEW_TO_FEATURIZER
        }
        state["view_dims"] = {
            block_label_to_view.get(name, name): dim for name, dim in featurizer._block_dims.items()
        }
    if featurizer._output_dim:
        state["output_dim"] = featurizer._output_dim
    return state
