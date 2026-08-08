"""Typed attachment of SparseGO ontology metadata on ``FeatureDataset``."""

from __future__ import annotations

from typing import TypedDict, cast

import numpy as np

from drevalpy.components._feature_dataset import FeatureDataset


class SparseGOOntologyMetadata(TypedDict):
    """Ontology graph metadata attached to a SparseGO ``FeatureDataset``."""

    layer_connections: list[np.ndarray]
    gene2id_mapping_ont: dict[str, int]
    ontology_gene_order: tuple[str, ...]
    gene_dim_input: int


def attach_sparsego_ontology_metadata(features: FeatureDataset, metadata: SparseGOOntologyMetadata) -> None:
    """Store SparseGO ontology metadata on a feature dataset instance.

    :param features: features.
    :param metadata: metadata.
    """
    vars(features)["_sparsego_ontology"] = metadata


def read_sparsego_ontology_metadata(features: FeatureDataset) -> SparseGOOntologyMetadata | None:
    """Return attached SparseGO ontology metadata when present.

    :param features: features.
    :returns: Result.
    """
    metadata = vars(features).get("_sparsego_ontology")
    if isinstance(metadata, dict):
        return cast(SparseGOOntologyMetadata, metadata)
    return None
