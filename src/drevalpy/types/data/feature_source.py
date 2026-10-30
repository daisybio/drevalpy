"""Feature access for featurizers: one :class:`FeatureSource` per entity side of a dataset.

A ``FeatureSource(dataset, side, entity_ids)`` reads the views of one side: sample views are
MuData modalities (plus ``pathway_features`` in ``response.obsm`` and ``tissue`` in the sample
metadata), drug views are ``response.varm`` keys (plus ``canonical_smiles`` in ``response.var``
and ``drug_graph`` in ``mdata.uns``). The side is a value, so one class serves both.

Pre-computed featurizer outputs ("stored variants") belong to the data layer too:
:meth:`FeatureSource.stored` reads them and :func:`store_variant` writes them. Each side keeps
a JSON variant registry in ``mdata.uns`` mapping a featurizer's storage key to
``{stored_key: hyperparameters}``; the matrix under ``stored_key`` is a MuData modality or,
failing that, ``response.varm`` (drug side) or ``response.obsm`` (sample side).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.dataset.aligned_fetch import aligned_fetch

Side = Literal["sample", "drug"]

_VARIANTS_UNS_KEYS: dict[Side, str] = {
    "sample": "sample_featurizer_variants",
    "drug": "drug_featurizer_variants",
}


def _variant_registry(mdata: Any, side: Side) -> dict[str, dict[str, dict[str, Any]]]:
    """Return the variant registry of *side*: ``{storage_key: {stored_key: hyperparameters}}``.

    :param mdata: MuData whose ``uns`` holds the registry.
    :param side: Side whose registry is read.
    :returns: The registry, empty when none is stored.
    """
    raw = mdata.uns.get(_VARIANTS_UNS_KEYS[side])
    if raw is None:
        return {}
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


def _variant_key(mdata: Any, side: Side, storage_key: str, params: Mapping[str, Any] | None) -> str | None:
    """Return the stored key of the variant of *storage_key* computed with *params*.

    :param mdata: MuData whose ``uns`` holds the variant registry.
    :param side: Side whose registry is read.
    :param storage_key: Featurizer storage key the variants are filed under.
    :param params: Hyperparameters the variant must have been computed with; ``None`` matches ``{}``.
    :returns: The stored key, or ``None`` when no variant matches.
    """
    target = dict(params or {})
    for key, variant_params in _variant_registry(mdata, side).get(storage_key, {}).items():
        if variant_params == target:
            return key
    return None


def store_variant(
    dataset: Dataset,
    side: Side,
    storage_key: str,
    matrix: np.ndarray,
    params: Mapping[str, Any] | None = None,
) -> str:
    """Store a pre-computed featurizer output so :meth:`FeatureSource.stored` finds it.

    The matrix goes to ``response.obsm`` (sample side, rows in ``response.obs_names`` order) or
    ``response.varm`` (drug side, rows in ``response.var_names`` order) under a new indexed key,
    which is filed under *storage_key* with *params* in the side's variant registry.

    :param dataset: Dataset that receives the matrix.
    :param side: Side of the featurizer that computed *matrix*.
    :param storage_key: Featurizer storage key the variant is filed under.
    :param matrix: Output with one row per sample (or drug) of the response modality.
    :param params: Hyperparameters *matrix* was computed with; ``None`` files it as ``{}``.
    :returns: The key the matrix is stored under.
    """
    mdata = dataset.mdata
    registry = _variant_registry(mdata, side)
    variants = registry.setdefault(storage_key, {})
    safe_key = storage_key.replace("[", "_").replace("]", "").replace(":", "_")
    stored_key = f"{safe_key}_{len(variants)}"
    response = mdata.mod["response"]
    if side == "drug":
        response.varm[stored_key] = matrix
    else:
        response.obsm[stored_key] = matrix
    variants[stored_key] = dict(params or {})
    mdata.uns[_VARIANTS_UNS_KEYS[side]] = json.dumps(registry)
    return stored_key


def _aligned(names: Any, data: Any, entity_ids: np.ndarray) -> np.ndarray:
    """Return the rows of *data* for *entity_ids*, NaN for IDs not in *names*.

    :param names: Row names of *data*.
    :param data: Stored matrix.
    :param entity_ids: Entity IDs in the order the rows are wanted.
    :returns: Float32 matrix with one row per ID.
    """
    import pandas as pd

    return aligned_fetch(
        pd.Index(names),
        np.asarray(entity_ids, dtype=str),
        np.asarray(data),
        strict=False,
        entity_label="entity",
    )


class FeatureSource:
    """The views of one entity side of a dataset, as featurizers read them."""

    def __init__(self, dataset: Dataset, side: Side, entity_ids: np.ndarray) -> None:
        """Wrap *dataset* for feature access on *side*.

        :param dataset: The backing dataset.
        :param side: ``"sample"`` or ``"drug"``.
        :param entity_ids: Entity IDs this source exposes.
        :raises ValueError: If *side* is not ``"sample"`` or ``"drug"``.
        """
        if side not in _VARIANTS_UNS_KEYS:
            msg = f"Unknown feature source side {side!r}. Available: drug, sample"
            raise ValueError(msg)
        self.side: Side = side
        self._dataset = dataset
        self._ids = np.asarray(entity_ids, dtype=str)

    @property
    def identifiers(self) -> np.ndarray:
        """All available entity IDs."""
        return self._ids

    @property
    def mdata(self) -> Any:
        """Underlying MuData object."""
        return self._dataset.mdata

    def get_metadata(self, key: str) -> Any:
        """Return arbitrary metadata from the underlying Dataset.

        :param key: ``mdata.uns`` key.
        :returns: The stored value.
        """
        return self._dataset.get_uns(key)

    def has_view(self, view: str) -> bool:
        """Return whether the dataset holds *view* for this side.

        Sample views: modality names, ``"pathway_features"`` (``response.obsm``) and ``"tissue"``
        (sample metadata). Drug views: ``response.varm`` keys, ``"canonical_smiles"`` (a
        ``response.var`` column) and ``"drug_graph"`` (graphs in ``mdata.uns``).

        :param view: View name.
        :returns: ``True`` when *view* can be read from this source.
        """
        if self.side == "sample":
            if view == "pathway_features":
                return "pathway_features" in self._dataset.response.obsm
            if view == "tissue":
                return "tissue" in self._dataset.sample_meta.columns
        else:
            if view == "canonical_smiles":
                return "canonical_smiles" in self._dataset.response.var.columns
            if view == "drug_graph":
                return "drug_graphs" in self._dataset.mdata.uns
        return self.get_feature_names(view) is not None

    def stored(self, key: str, params: Mapping[str, Any] | None, ids: np.ndarray) -> np.ndarray | None:
        """Return a pre-computed featurizer output for *ids*, if the dataset stores one.

        :param key: Featurizer storage key the variants are filed under.
        :param params: Hyperparameters the stored variant must match; ``None`` matches the
            parameter-free variant.
        :param ids: Entity IDs to align the stored rows to.
        :returns: Float32 matrix with one row per ID (NaN for IDs the variant lacks), or ``None``
            when the source has no MuData or no variant matches.
        """
        mdata = self.mdata
        if mdata is None:
            return None
        stored_key = _variant_key(mdata, self.side, key, params)
        if stored_key is None:
            return None
        if stored_key in mdata.mod:
            modality = mdata.mod[stored_key]
            return _aligned(modality.obs_names, modality.X, ids)
        response = mdata.mod.get("response")
        if response is None:
            return None
        if self.side == "drug":
            if response.varm is None or stored_key not in response.varm:
                return None
            return _aligned(response.var_names, response.varm[stored_key], ids)
        if response.obsm is None or stored_key not in response.obsm:
            return None
        return _aligned(response.obs_names, response.obsm[stored_key], ids)

    def get_view_matrix(self, view: str, entity_ids: np.ndarray) -> np.ndarray:
        """Return the ``(len(entity_ids), n_features)`` matrix of a dense numeric view.

        :param view: View name.
        :param entity_ids: Entity IDs, in row order.
        :returns: One row per ID.
        """
        if self.side == "sample":
            return self._dataset.get_sample_features(view, entity_ids)
        return self._dataset.get_drug_features(view, entity_ids)

    def get_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return the ordered feature names of a view.

        :param view: View name.
        :returns: Column names, or ``None`` when the view has none.
        """
        if self.side == "sample":
            return self._dataset.get_sample_feature_names(view)
        return self._dataset.get_drug_feature_names(view)

    def get_entity_view(self, entity_id: str, view: str) -> Any:
        """Return the raw value of *view* for one entity.

        ``"tissue"`` (sample) and ``"drug_graph"`` (drug) give the tissue label and the graph
        dict, ``None`` when the dataset lacks them; other views give the entity's feature vector
        (``None`` for a drug view the dataset lacks).

        :param entity_id: Entity ID.
        :param view: View name.
        :returns: The per-entity value.
        """
        ids = np.array([entity_id])
        if self.side == "sample":
            if view == "tissue":
                try:
                    return self._dataset.get_tissue(ids)[0]
                except KeyError:
                    return None
            return self._dataset.get_sample_features(view, ids)[0]
        if view == "drug_graph":
            graphs = self._dataset.get_drug_graphs(ids)
            return graphs[0] if graphs else None
        try:
            return self._dataset.get_drug_features(view, ids)[0]
        except KeyError:
            return None
