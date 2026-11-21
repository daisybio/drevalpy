"""Feature access for featurizers: one :class:`FeatureSource` per entity side of a dataset.

A ``FeatureSource(dataset, side)`` is the only way a featurizer reads a dataset. Its views by
side:

- sample: dense views are MuData modalities (looked up through
  :func:`~drevalpy.types.data.views.find_view`) or, when no modality has the name,
  ``response.obsm`` keys such as ``pathway_features``; :meth:`FeatureSource.tissue` reads the
  ``tissue`` column of the sample metadata.
- drug: dense views are ``response.varm`` keys; :meth:`FeatureSource.smiles` reads the
  ``canonical_smiles`` column of ``response.var`` and :meth:`FeatureSource.graphs` the graphs in
  ``mdata.uns["drug_graphs"]``.

Dense views come back as float32 rows aligned to the requested IDs, NaN rows for unknown IDs.

A view written by :func:`drevalpy.data.precompute.precompute` has the settings it was computed
with recorded under :data:`~drevalpy.types.data.views.PRECOMPUTED_VIEWS_KEY` in ``mdata.uns``;
:meth:`FeatureSource.view_settings` reads them.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

import numpy as np

from drevalpy.errors import InvalidInputError
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.dataset._aligned_fetch import aligned_fetch
from drevalpy.types.data.dataset._anndata import to_dense
from drevalpy.types.data.views import (
    DRUG_GRAPH_VIEW,
    DRUG_GRAPHS_KEY,
    EXPRESSION_UNIT_KEY,
    PRECOMPUTED_VIEWS_KEY,
    SMILES_COLUMN,
    TISSUE_IDENTIFIER,
    find_view,
    public_view_name,
)


class Side(StrEnum):
    """The entity side a featurizer reads: the samples or the drugs."""

    SAMPLE = "sample"
    DRUG = "drug"


def precomputed_views(mdata: Any) -> dict[str, dict[str, dict[str, Any]]]:
    """Return the recorded settings of the precomputed views of *mdata*: ``{side: {view: settings}}``.

    :param mdata: MuData whose ``uns`` holds the record.
    :returns: The record, empty when nothing is recorded.
    """
    raw = mdata.uns.get(PRECOMPUTED_VIEWS_KEY)
    if raw is None:
        return {}
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


class FeatureSource:
    """The views of one entity side of a dataset, as featurizers read them."""

    def __init__(self, dataset: Dataset, side: Side | str) -> None:
        """Wrap *dataset* for feature access on *side*.

        :param dataset: The backing dataset.
        :param side: The side to read.
        :raises ValueError: If *side* is not a side.
        """
        try:
            self.side = Side(side)
        except ValueError as err:
            available = ", ".join(sorted(s.value for s in Side))
            msg = f"Unknown feature source side {side!r}. Available: {available}"
            raise ValueError(msg) from err
        self._dataset = dataset

    def _require_side(self, expected: Side) -> None:
        """Reject calls made on the wrong entity side."""
        if self.side != expected:
            msg = f"This accessor requires a {expected.value!r} FeatureSource, got {self.side.value!r}."
            raise ValueError(msg)

    def _dense(self, name: str) -> tuple[Any, Any, Any] | None:
        """Return ``(index, matrix, raw_feature_names)`` for dense view *name*, or ``None`` if absent."""
        mdata = self._dataset.mdata
        response = self._dataset.response
        if self.side == Side.DRUG:
            varm = response.varm or {}
            if name not in varm:
                return None
            data = varm[name]
            names = data.columns if hasattr(data, "columns") else (range(data.shape[1]) if data.ndim > 1 else None)
            return response.var_names, data, names
        if name != "response":
            stored = find_view(name, mdata.mod)
            if stored is not None:
                mod = mdata.mod[stored]
                return mod.obs_names, mod.X, mod.var_names
        if name in response.obsm:
            return response.obs_names, response.obsm[name], None
        return None

    def _available_dense_views(self) -> list[str]:
        """Return the sorted public names of every dense view on this side."""
        response = self._dataset.response
        if self.side == Side.DRUG:
            return sorted(response.varm or ())
        modalities = (public_view_name(key) for key in self._dataset.mdata.mod if key != "response")
        return sorted({*modalities, *response.obsm})

    def __contains__(self, name: str) -> bool:
        """Return whether the dataset holds view *name* for this side.

        Besides the dense views, the sample side has ``"tissue"`` (sample metadata) and the drug
        side ``"canonical_smiles"`` (a ``response.var`` column) and ``"drug_graph"`` (graphs in
        ``mdata.uns``).

        :param name: View name.
        :returns: ``True`` when *name* can be read from this source.
        """
        if self.side == Side.SAMPLE and name == TISSUE_IDENTIFIER:
            return TISSUE_IDENTIFIER in self._dataset.sample_meta.columns
        if self.side == Side.DRUG and name == SMILES_COLUMN:
            return SMILES_COLUMN in self._dataset.response.var.columns
        if self.side == Side.DRUG and name == DRUG_GRAPH_VIEW:
            return DRUG_GRAPHS_KEY in self._dataset.mdata.uns
        return self._dense(name) is not None

    def view(self, name: str, ids: np.ndarray) -> np.ndarray:
        """Return the ``(len(ids), n_features)`` float32 matrix of dense view *name*.

        :param name: View name.
        :param ids: Entity IDs, in row order; an ID the view lacks gets a NaN row and a warning.
        :returns: One row per ID.
        :raises KeyError: If this side has no dense view *name*.
        :raises InvalidInputError: If the drug view is an object array of per-drug arrays (such as ragged
            MolGNet node features), which has no matrix form.
        """
        import pandas as pd

        found = self._dense(name)
        if found is None:
            raise KeyError(
                f"{self.side.capitalize()} view '{name}' not found. Available: {self._available_dense_views()}"
            )
        index, data, _ = found
        data = np.asarray(to_dense(data))
        if data.dtype == object:
            try:
                data = data.astype(np.float32)
            except (ValueError, TypeError) as err:
                msg = (
                    f"View '{name}' is an object array (one array per entity, such as ragged per-atom node "
                    "features) and cannot be read as a (n_entities, n_features) matrix. A .h5mu file cannot "
                    "store such a view; remove the key so the featurizer computes it from the SMILES instead."
                )
                raise InvalidInputError(msg) from err
        return aligned_fetch(pd.Index(index), np.asarray(ids, dtype=str), data, strict=False, entity_label=self.side)

    def feature_names(self, name: str) -> tuple[str, ...] | None:
        """Return the ordered feature names of dense view *name*.

        :param name: View name.
        :returns: Column names (positions as strings for an unnamed drug view), or ``None`` when the view is
            absent or, as for ``response.obsm`` views, has no names.
        """
        found = self._dense(name)
        if found is None or found[2] is None:
            return None
        return tuple(str(col) for col in found[2])

    def view_settings(self, name: str) -> dict[str, Any] | None:
        """Return the settings view *name* was precomputed with, or ``None`` when none are recorded.

        :param name: View name.
        :returns: The recorded settings; ``None`` for a view the dataset ships without a record.
        """
        return precomputed_views(self._dataset.mdata).get(self.side, {}).get(name)

    def expression_unit(self, name: str) -> str | None:
        """Return the ``expression_unit`` recorded in the ``uns`` of sample view *name*.

        Requires a sample source (raises ``ValueError`` on a drug source).

        :param name: View name.
        :returns: The unit, or ``None`` when the view is absent or records none.
        """
        self._require_side(Side.SAMPLE)
        if name == "response":
            return None
        stored = find_view(name, self._dataset.mdata.mod)
        if stored is None:
            return None
        unit = self._dataset.mdata.mod[stored].uns.get(EXPRESSION_UNIT_KEY)
        return None if unit is None else str(unit)

    def smiles(self, ids: np.ndarray) -> np.ndarray | None:
        """Return the canonical SMILES of the drugs *ids*.

        Requires a drug source (raises ``ValueError`` on a sample source).

        :param ids: Drug IDs.
        :returns: One SMILES string per ID (NaN for unknown drugs), or ``None`` when the dataset has none.
        """
        self._require_side(Side.DRUG)
        var = self._dataset.response.var
        if SMILES_COLUMN not in var.columns:
            return None
        return var[SMILES_COLUMN].reindex(np.asarray(ids, dtype=str)).to_numpy()

    def graphs(self, ids: np.ndarray) -> list[dict[str, np.ndarray] | None] | None:
        """Return the stored graph of every drug in *ids*.

        Each graph dict has the keys ``"x"``, ``"edge_index"`` and ``"edge_attr"``. Requires a drug
        source (raises ``ValueError`` on a sample source).

        :param ids: Drug IDs.
        :returns: One graph per ID (``None`` for a drug without one), or ``None`` when none are stored.
        """
        self._require_side(Side.DRUG)
        graphs = self._dataset.mdata.uns.get(DRUG_GRAPHS_KEY)
        if graphs is None:
            return None
        return [graphs.get(drug_id) for drug_id in np.asarray(ids, dtype=str)]

    def tissue(self, ids: np.ndarray) -> np.ndarray | None:
        """Return the tissue label of every sample in *ids*.

        Requires a sample source (raises ``ValueError`` on a drug source).

        :param ids: Sample IDs.
        :returns: One label per ID (NaN for unknown samples), or ``None`` when the dataset has no tissues.
        """
        self._require_side(Side.SAMPLE)
        if TISSUE_IDENTIFIER not in self._dataset.sample_meta.columns:
            return None
        return self._dataset.get_tissue(ids)

    def uns(self, key: str) -> Any:
        """Return the dataset-level auxiliary value stored under *key* in ``mdata.uns``.

        :param key: ``mdata.uns`` key, such as ``"pathways_gmt"`` or ``"sparsego"``.
        :returns: The stored value, or ``None`` when the dataset has none.
        """
        return self._dataset.mdata.uns.get(key)
