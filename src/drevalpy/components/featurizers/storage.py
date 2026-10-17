"""Featurizer variant storage: the MuData helpers and the mixin built on them."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from drevalpy.types.data.feature_source import FeatureSource

VARIANTS_UNS_KEY_CELL_LINE = "cell_line_featurizer_variants"
VARIANTS_UNS_KEY_DRUG = "drug_featurizer_variants"


def _variants_key_for_side(side: str) -> str:
    """Return the uns key for the given side."""
    if side == "drug":
        return VARIANTS_UNS_KEY_DRUG
    return VARIANTS_UNS_KEY_CELL_LINE


def _variant_registry(mdata, side: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Read the featurizer variant registry from mdata.uns.

    Format: {storage_key: {mudata_key: params_dict, ...}, ...}
    """
    uns_key = _variants_key_for_side(side)
    raw = mdata.uns.get(uns_key)
    if raw is None:
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def _write_variant_registry(mdata, registry: dict[str, dict[str, dict[str, Any]]], side: str) -> None:
    """Write the featurizer variant registry to mdata.uns."""
    uns_key = _variants_key_for_side(side)
    mdata.uns[uns_key] = json.dumps(registry)


def find_variant_key(
    mdata,
    storage_key: str,
    hyperparameters: dict[str, Any] | None = None,
    *,
    side: str = "cell_line",
) -> str | None:
    """Find the MuData storage key for a featurizer variant matching the given HPs.

    :param mdata: MuData object.
    :param storage_key: Base storage key for the featurizer.
    :param hyperparameters: HP dict to match against stored variants.
    :param side: Entity side ("cell_line" or "drug").
    :returns: The actual MuData key, or None if not found.
    """
    registry = _variant_registry(mdata, side)
    variants = registry.get(storage_key, {})

    target_params = hyperparameters or {}
    for key, params in variants.items():
        if params == target_params:
            return key
    return None


def list_variants(mdata, storage_key: str, *, side: str = "cell_line") -> dict[str, dict[str, Any]]:
    """List all stored HP variants for a featurizer.

    :param mdata: MuData object.
    :param storage_key: Base storage key for the featurizer.
    :param side: Entity side ("cell_line" or "drug").
    :returns: Dict of {mudata_key: params_dict}.
    """
    registry = _variant_registry(mdata, side)
    return registry.get(storage_key, {})


def register_variant(
    mdata,
    storage_key: str,
    actual_key: str,
    hyperparameters: dict[str, Any] | None = None,
    *,
    side: str = "cell_line",
) -> None:
    """Register a new variant in the featurizer variant registry.

    :param mdata: MuData object.
    :param storage_key: Base storage key for the featurizer.
    :param actual_key: The actual MuData key where data is stored.
    :param hyperparameters: HP settings for this variant.
    :param side: Entity side ("cell_line" or "drug").
    """
    registry = _variant_registry(mdata, side)
    variants = registry.setdefault(storage_key, {})
    variants[actual_key] = hyperparameters or {}
    _write_variant_registry(mdata, registry, side)


def make_variant_key(storage_key: str, index: int) -> str:
    """Generate the indexed storage key for a variant.

    :param storage_key: Base featurizer storage key.
    :param index: Variant index.
    :returns: Key like "pca_0".
    """
    safe_key = storage_key.replace("[", "_").replace("]", "").replace(":", "_")
    return f"{safe_key}_{index}"


def next_variant_index(mdata, storage_key: str, *, side: str = "cell_line") -> int:
    """Return the next available variant index."""
    variants = list_variants(mdata, storage_key, side=side)
    return len(variants)


def fetch_from_modality(mdata, modality: str, entity_ids: np.ndarray) -> np.ndarray | None:
    """Fetch a matrix from a MuData modality, aligned to entity_ids.

    :returns: Float array or None if modality doesn't exist.
    """
    import pandas as pd

    if modality not in mdata.mod:
        return None
    mod = mdata.mod[modality]

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
    import pandas as pd

    response = mdata.mod.get("response")
    if response is None or response.varm is None or key not in response.varm:
        return None

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
    import pandas as pd

    response = mdata.mod.get("response")
    if response is None or response.obsm is None or key not in response.obsm:
        return None

    obsm_data = np.asarray(response.obsm[key])
    idx = pd.Index(response.obs_names)
    positions = idx.get_indexer(entity_ids)
    found = positions >= 0
    if not found.all():
        result = np.full((len(entity_ids), obsm_data.shape[1]), np.nan, dtype=np.float32)
        result[found] = obsm_data[positions[found]].astype(np.float32)
        return result
    return obsm_data[positions].astype(np.float32)


class FeaturizerStorageMixin:
    """Read and write pre-computed featurizer matrices in a MuData store.

    Extracted from ``Featurizer``, which it is mixed back into: these five methods
    were the one cohesive cluster on that class, and they reach nothing beyond the
    two class attributes declared below. Keeping them here means a featurizer that
    only ever computes on the fly still gets the storage protocol, without the
    protocol being tangled into the fit/transform contract.
    """

    #: Base key the variant registry files this featurizer's matrices under.
    #: Registration defaults it to the registry name.
    storage_key: ClassVar[str] = ""
    #: Entity side, stamped on by the registry the featurizer is registered in.
    side: ClassVar[str] = ""

    def fetch(
        self, mdata: Any, entity_ids: np.ndarray, hyperparameters: dict[str, Any] | None = None
    ) -> np.ndarray | None:
        """Fetch pre-computed representations from MuData for the given HPs.

        :param mdata: MuData object.
        :param entity_ids: Entity IDs to fetch for.
        :param hyperparameters: HP setting to match. None matches default (empty params).
        :returns: Feature matrix or None if not pre-computed for these HPs.
        """
        key = find_variant_key(mdata, self.storage_key, hyperparameters, side=self.side)
        if key is None:
            return None
        return self._fetch_by_key(mdata, key, entity_ids)

    def fetch_precomputed(
        self,
        source: FeatureSource,
        entity_ids: np.ndarray,
        hyperparameters: dict[str, Any] | None = None,
    ) -> np.ndarray | None:
        """Fetch a pre-computed matrix through a feature source, if it carries one.

        Every featurizer that can be pre-computed by ``Dataset.precompute()`` opens
        ``_fit`` and ``_transform`` by asking whether the work is already done. Only
        dataset-backed sources expose ``mdata``, so the question has to tolerate a
        source without one.

        :param source: Feature source, which may or may not be MuData-backed.
        :param entity_ids: Entity IDs to align the stored matrix to.
        :param hyperparameters: HP setting to match; ``None`` matches the default variant.
        :returns: Stored feature matrix, or ``None`` when nothing matches.
        """
        mdata = getattr(source, "mdata", None)
        if mdata is None:
            return None
        return self.fetch(mdata, entity_ids, hyperparameters)

    def _fetch_by_key(self, mdata: Any, key: str, entity_ids: np.ndarray) -> np.ndarray | None:
        """Fetch data from MuData by resolved key. Override for custom storage.

        :param mdata: MuData object.
        :param key: Resolved storage key (e.g., "pca_expression_0").
        :param entity_ids: Entity IDs to align to.
        :returns: Feature matrix or None.
        """
        result = fetch_from_modality(mdata, key, entity_ids)
        if result is not None:
            return result
        if self.side == "drug":
            return fetch_from_varm(mdata, key, entity_ids)
        return fetch_from_obsm(mdata, key, entity_ids)

    def store(
        self, mdata: Any, entity_ids: np.ndarray, data: np.ndarray, hyperparameters: dict[str, Any] | None = None
    ) -> None:
        """Store computed representations into MuData with HP metadata.

        :param mdata: MuData object.
        :param entity_ids: Entity IDs the data is aligned to.
        :param data: Feature matrix to store.
        :param hyperparameters: HP settings this data was computed with.
        """
        index = next_variant_index(mdata, self.storage_key, side=self.side)
        actual_key = make_variant_key(self.storage_key, index)
        self._store_by_key(mdata, actual_key, entity_ids, data)
        register_variant(mdata, self.storage_key, actual_key, hyperparameters, side=self.side)

    def _store_by_key(self, mdata: Any, key: str, entity_ids: np.ndarray, data: np.ndarray) -> None:
        """Write data to MuData under the given key. Override for custom storage.

        Default stores in response.obsm for cell-line-side, varm for drug-side.

        :param mdata: MuData object.
        :param key: Storage key.
        :param entity_ids: Entity IDs.
        :param data: Data matrix.
        """
        response = mdata.mod["response"]
        if self.side == "drug":
            response.varm[key] = data
        else:
            response.obsm[key] = data

    @classmethod
    def list_stored_variants(cls, mdata: Any) -> dict[str, dict[str, Any]]:
        """Return available pre-computed HP settings for this featurizer.

        :param mdata: MuData object.
        :returns: Dict of {mudata_key: params_dict}.
        """
        return list_variants(mdata, cls.storage_key, side=cls.side)
