"""Typed access to SparseGO ontology metadata via ``FeatureSource``."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from drevalpy.components.feature_source import FeatureSource


class SparseGOOntologyMetadata(TypedDict):
    """Ontology graph metadata attached to a SparseGO ``FeatureSource``."""

    layer_connections: list[np.ndarray]
    gene2id_mapping_ont: dict[str, int]
    ontology_gene_order: tuple[str, ...]
    gene_dim_input: int


def read_sparsego_ontology_metadata(source: FeatureSource) -> SparseGOOntologyMetadata | None:
    """Return SparseGO ontology metadata from a feature source.

    :param source: Feature source with metadata access.
    :returns: Result.
    """
    metadata: Any = source.get_metadata("sparsego_ontology")
    if isinstance(metadata, dict):
        return {
            "layer_connections": metadata["layer_connections"],
            "gene2id_mapping_ont": metadata["gene2id_mapping_ont"],
            "ontology_gene_order": metadata["ontology_gene_order"],
            "gene_dim_input": metadata["gene_dim_input"],
        }
    return None
