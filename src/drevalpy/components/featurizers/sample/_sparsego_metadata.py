"""Typed access to SparseGO ontology metadata via ``FeatureSource``."""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from drevalpy.types.data.feature_source import FeatureSource


class SparseGOOntologyMetadata(TypedDict):
    """Ontology graph metadata attached to a SparseGO ``FeatureSource``."""

    layer_connections: list[np.ndarray]
    gene2id_mapping_ont: dict[str, int]
    ontology_gene_order: tuple[str, ...]
    gene_dim_input: int


def read_sparsego_ontology_metadata(source: FeatureSource) -> SparseGOOntologyMetadata | None:
    """Return SparseGO ontology metadata from a feature source.

    :param source: Feature source with metadata access.
    :returns: Typed metadata, or ``None`` when the source has no usable SparseGO ontology entry.
    """
    metadata: Any = None
    for key in ("sparsego_ontology", "sparsego"):
        try:
            metadata = source.get_metadata(key)
        except (KeyError, AttributeError):
            continue
        if metadata is not None:
            break
    if isinstance(metadata, dict) and (
        "layer_connections" in metadata or "ontology_file" in metadata or "ontology" in metadata
    ):
        from drevalpy.components.predictors.literature.sparsego.utils import resolve_ontology_metadata

        connections, mapping, order = resolve_ontology_metadata(metadata)
        return {
            "layer_connections": connections,
            "gene2id_mapping_ont": mapping,
            "ontology_gene_order": tuple(order),
            "gene_dim_input": len(order),
        }
    return None
