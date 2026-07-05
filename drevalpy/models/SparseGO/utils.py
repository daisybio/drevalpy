"""Compatibility re-exports for SparseGO utilities."""

from drevalpy.components.predictors.literature.impl.sparsego.utils import (
    create_index,
    load_mapping,
    load_ontology,
    pairs_in_layers,
    sort_pairs,
)

__all__ = [
    "create_index",
    "load_mapping",
    "load_ontology",
    "pairs_in_layers",
    "sort_pairs",
]
