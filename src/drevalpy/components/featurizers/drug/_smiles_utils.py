"""Shared utilities for accessing SMILES strings from a FeatureSource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy.types.data.feature_source import FeatureSource

if TYPE_CHECKING:
    import pandas as pd


def get_smiles_for_entities(source: FeatureSource, entity_ids: np.ndarray) -> pd.Series | None:
    """Get canonical SMILES indexed by entity_ids from the dataset.

    :param source: Feature source backed by a MuData object.
    :param entity_ids: Drug identifiers to retrieve SMILES for.
    :returns: Series of SMILES strings indexed by entity_ids, or None if unavailable.
    """
    mdata = getattr(source, "mdata", None)
    if mdata is None:
        return None
    response = mdata.mod["response"]
    if "canonical_smiles" not in response.var.columns:
        return None
    smiles = response.var["canonical_smiles"]
    return smiles.reindex(entity_ids)


def as_pyg_graph(graph: Any) -> Any:
    """Convert a portable h5mu graph mapping to a PyG payload on demand.

    :param graph: A graph mapping, existing PyG payload, or None.
    :returns: The corresponding graph without modifying stored arrays.
    """
    if not isinstance(graph, dict):
        return graph
    import torch
    from torch_geometric.data import Data

    return Data(
        **{
            key: torch.as_tensor(value, dtype=torch.long if key == "edge_index" else torch.float32)
            for key, value in graph.items()
        }
    )
