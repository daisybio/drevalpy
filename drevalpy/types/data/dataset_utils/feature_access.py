"""Standalone functions for accessing cell-line and drug features from MuData."""

from __future__ import annotations

import mudata as md
import numpy as np
import pandas as pd

from .aligned_fetch import _aligned_fetch


def get_cell_line_features(mdata: md.MuData, modality: str, ids: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Get a feature matrix for the specified cell lines from a modality.

    Args:
        mdata: The MuData object containing the dataset.
        modality: Name of the modality (e.g. "gene_expression").
        ids: 1-D array of cell line IDs to retrieve.
        strict: If True, raise KeyError for missing IDs instead of warning.

    Returns:
        Float32 array of shape (len(ids), n_features), rows aligned to *ids*.
    """
    ids = np.asarray(ids, dtype=str)

    if modality == "pathway_features":
        return _get_obsm_features(mdata, "pathway_features", ids, strict=strict)

    if modality not in mdata.mod:
        raise KeyError(f"Modality '{modality}' not found. Available: {list(mdata.mod.keys())}")

    adata = mdata.mod[modality]
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return _aligned_fetch(pd.Index(adata.obs_names), ids, np.asarray(x), strict=strict, entity_label="cell line")


def _get_obsm_features(mdata: md.MuData, key: str, ids: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Retrieve cell-line features stored in response.obsm."""
    response = mdata.mod["response"]
    if key not in response.obsm:
        raise KeyError(f"obsm key '{key}' not found in response modality.")

    obsm_data = np.asarray(response.obsm[key])
    return _aligned_fetch(pd.Index(response.obs_names), ids, obsm_data, strict=strict, entity_label="cell line")


def get_cell_line_feature_names(mdata: md.MuData, view: str) -> tuple[str, ...] | None:
    """Return the feature (column) names for a cell-line view.

    Args:
        mdata: The MuData object containing the dataset.
        view: Name of the modality.

    Returns:
        Tuple of feature names, or None if names are unavailable.
    """
    if view == "pathway_features":
        return None
    if view not in mdata.mod:
        return None
    return tuple(mdata.mod[view].var_names)


def _resolve_varm_key(mdata: md.MuData, name: str) -> str | None:
    """Resolve a varm key by exact match or prefix match (name:variant)."""
    varm = mdata.mod["response"].varm
    if varm is None:
        return None
    if name in varm:
        return name
    for key in varm.keys():
        if key.startswith(name + ":"):
            return key
    return None


def available_drug_views(mdata: md.MuData) -> list[str]:
    """Sorted list of drug feature varm keys."""
    response = mdata.mod["response"]
    if response.varm is None:
        return []
    return sorted(response.varm.keys())


def get_drug_features(mdata: md.MuData, name: str, ids: np.ndarray, *, strict: bool = False) -> np.ndarray:
    """Get a drug feature matrix from response.varm, aligned to given IDs.

    Args:
        mdata: The MuData object containing the dataset.
        name: Key in ``response.varm`` (e.g. "chemberta", "morgan_fingerprint").
        ids: 1-D array of drug (PubChem) IDs.
        strict: If True, raise KeyError for missing IDs instead of warning.

    Returns:
        Float32 array of shape (len(ids), n_features), rows aligned to *ids*.
    """
    varm_key = _resolve_varm_key(mdata, name)
    if varm_key is None:
        raise KeyError(f"Drug feature '{name}' not found. Available varm keys: {available_drug_views(mdata)}")

    response = mdata.mod["response"]
    ids = np.asarray(ids, dtype=str)
    varm_data = np.asarray(response.varm[varm_key])
    return _aligned_fetch(pd.Index(response.var_names), ids, varm_data, strict=strict, entity_label="drug")


def get_drug_feature_names(mdata: md.MuData, view: str) -> tuple[str, ...] | None:
    """Return the feature (column) names for a drug view stored in response.varm.

    Args:
        mdata: The MuData object containing the dataset.
        view: Drug view name (e.g. "chemberta", "morgan_fingerprint").

    Returns:
        Tuple of column name strings, or None if the view does not exist.
    """
    varm_key = _resolve_varm_key(mdata, view)
    if varm_key is None:
        return None
    varm_data = mdata.mod["response"].varm[varm_key]
    if hasattr(varm_data, "columns"):
        return tuple(varm_data.columns.astype(str))
    return tuple(str(i) for i in range(varm_data.shape[1]))


def get_drug_graphs(mdata: md.MuData, ids: np.ndarray) -> list[dict[str, np.ndarray] | None]:
    """Get PyTorch Geometric graph data for the specified drugs.

    Each graph dict has keys "x", "edge_index", "edge_attr" with numpy arrays.
    Returns None for drugs without a stored graph.

    Args:
        mdata: The MuData object containing the dataset.
        ids: 1-D array of drug (PubChem) IDs.

    Returns:
        List of graph dicts (or None) aligned to *ids*.
    """
    if "drug_graphs" not in mdata.uns:
        raise KeyError("'drug_graphs' not found in mdata.uns.")

    ids = np.asarray(ids, dtype=str)
    graphs = mdata.uns["drug_graphs"]
    return [graphs.get(drug_id) for drug_id in ids]
