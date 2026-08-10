"""Featurizer variant storage helpers for MuData."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

VARIANTS_UNS_KEY = "featurizer_variants"


def _variant_registry(mdata) -> dict[str, list[dict[str, Any]]]:
    """Read the featurizer variant registry from mdata.uns."""
    raw = mdata.uns.get(VARIANTS_UNS_KEY)
    if raw is None:
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def _write_variant_registry(mdata, registry: dict[str, list[dict[str, Any]]]) -> None:
    """Write the featurizer variant registry to mdata.uns."""
    mdata.uns[VARIANTS_UNS_KEY] = json.dumps(registry)


def find_variant_key(
    mdata,
    storage_key: str,
    hyperparameters: dict[str, Any] | None = None,
) -> str | None:
    """Find the MuData storage key for a featurizer variant matching the given HPs.

    :param mdata: MuData object.
    :param storage_key: Base storage key for the featurizer.
    :param hyperparameters: HP dict to match against stored variants.
    :returns: The actual MuData key, or None if not found.
    """
    registry = _variant_registry(mdata)
    variants = registry.get(storage_key, [])

    if not variants:
        return None

    target_params = hyperparameters or {}
    for variant in variants:
        if variant.get("params", {}) == target_params:
            return variant["key"]
    return None


def list_variants(mdata, storage_key: str) -> list[dict[str, Any]]:
    """List all stored HP variants for a featurizer.

    :param mdata: MuData object.
    :param storage_key: Base storage key for the featurizer.
    :returns: List of variant dicts with "params" and "key".
    """
    registry = _variant_registry(mdata)
    return registry.get(storage_key, [])


def register_variant(
    mdata,
    storage_key: str,
    actual_key: str,
    hyperparameters: dict[str, Any] | None = None,
) -> None:
    """Register a new variant in the featurizer variant registry.

    :param mdata: MuData object.
    :param storage_key: Base storage key for the featurizer.
    :param actual_key: The actual MuData key where data is stored.
    :param hyperparameters: HP settings for this variant.
    """
    registry = _variant_registry(mdata)
    variants = registry.setdefault(storage_key, [])
    variants.append(
        {
            "params": hyperparameters or {},
            "key": actual_key,
        }
    )
    _write_variant_registry(mdata, registry)


def make_variant_key(storage_key: str, index: int) -> str:
    """Generate the indexed storage key for a variant.

    :param storage_key: Base featurizer storage key.
    :param index: Variant index.
    :returns: Key like "pca_expression_0".
    """
    safe_key = storage_key.replace("[", "_").replace("]", "").replace(":", "_")
    return f"{safe_key}_{index}"


def next_variant_index(mdata, storage_key: str) -> int:
    """Return the next available variant index."""
    variants = list_variants(mdata, storage_key)
    return len(variants)


def fetch_from_modality(mdata, modality: str, entity_ids: np.ndarray) -> np.ndarray | None:
    """Fetch a matrix from a MuData modality, aligned to entity_ids.

    :returns: Float array or None if modality doesn't exist.
    """
    if modality not in mdata.mod:
        return None
    mod = mdata.mod[modality]
    import pandas as pd

    idx = pd.Index(mod.obs_names)
    positions = idx.get_indexer(entity_ids)
    found = positions >= 0
    if not found.all():
        result = np.full((len(entity_ids), mod.X.shape[1]), np.nan, dtype=np.float32)
        result[found] = np.asarray(mod.X[positions[found]], dtype=np.float32)
        return result
    return np.asarray(mod.X[positions], dtype=np.float32)


def fetch_from_varm(mdata, key: str, entity_ids: np.ndarray) -> np.ndarray | None:
    """Fetch a matrix from response.varm, aligned to entity_ids.

    :returns: Float array or None if key doesn't exist.
    """
    response = mdata.mod.get("response")
    if response is None or response.varm is None or key not in response.varm:
        return None
    import pandas as pd

    varm_data = np.asarray(response.varm[key])
    idx = pd.Index(response.var_names)
    positions = idx.get_indexer(entity_ids)
    found = positions >= 0
    if not found.all():
        result = np.full((len(entity_ids), varm_data.shape[1]), np.nan, dtype=np.float32)
        result[found] = varm_data[positions[found]].astype(np.float32)
        return result
    return varm_data[positions].astype(np.float32)


def fetch_from_obsm(mdata, key: str, entity_ids: np.ndarray) -> np.ndarray | None:
    """Fetch a matrix from response.obsm, aligned to entity_ids.

    :returns: Float array or None if key doesn't exist.
    """
    response = mdata.mod.get("response")
    if response is None or response.obsm is None or key not in response.obsm:
        return None
    import pandas as pd

    obsm_data = np.asarray(response.obsm[key])
    idx = pd.Index(response.obs_names)
    positions = idx.get_indexer(entity_ids)
    found = positions >= 0
    if not found.all():
        result = np.full((len(entity_ids), obsm_data.shape[1]), np.nan, dtype=np.float32)
        result[found] = obsm_data[positions[found]].astype(np.float32)
        return result
    return obsm_data[positions].astype(np.float32)
