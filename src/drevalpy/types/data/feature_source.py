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
with recorded under :data:`PRECOMPUTED_VIEWS_KEY` in ``mdata.uns``;
:meth:`FeatureSource.view_settings` reads them.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy.errors import InvalidInputError
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.dataset._anndata import to_dense
from drevalpy.types.data.dataset.aligned_fetch import aligned_fetch
from drevalpy.types.data.views import (
    DRUG_GRAPH_VIEW,
    DRUG_GRAPHS_KEY,
    EXPRESSION_UNIT_KEY,
    SMILES_COLUMN,
    TISSUE_IDENTIFIER,
    find_view,
    public_view_name,
)

if TYPE_CHECKING:
    import pandas as pd


class Side(StrEnum):
    """The entity side a featurizer reads: the samples or the drugs."""

    SAMPLE = "sample"
    DRUG = "drug"


#: ``mdata.uns`` key of the settings of precomputed views, a JSON object ``{side: {view: settings}}``.
PRECOMPUTED_VIEWS_KEY = "precomputed_views"


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

    def __init__(self, dataset: Dataset, side: Side) -> None:
        """Wrap *dataset* for feature access on *side*.

        :param dataset: The backing dataset.
        :param side: The side to read.
        :raises ValueError: If *side* is not a side.
        """
        if side not in tuple(Side):
            msg = f"Unknown feature source side {side!r}. Available: drug, sample"
            raise ValueError(msg)
        self.side = Side(side)
        self._dataset = dataset

    def _dense(self, name: str) -> tuple[Any, np.ndarray] | None:
        """Return the entity index and the matrix of dense view *name*, or ``None`` if this side lacks it."""
        mdata = self._dataset.mdata
        response = self._dataset.response
        if self.side == "drug":
            if name not in (response.varm or {}):
                return None
            return response.var_names, response.varm[name]
        stored = find_view(name, mdata.mod)
        if stored is not None:
            return mdata.mod[stored].obs_names, mdata.mod[stored].X
        if name in response.obsm:
            return response.obs_names, response.obsm[name]
        return None

    def has_view(self, name: str) -> bool:
        """Return whether the dataset holds view *name* for this side.

        Besides the dense views, the sample side has ``"tissue"`` (sample metadata) and the drug
        side ``"canonical_smiles"`` (a ``response.var`` column) and ``"drug_graph"`` (graphs in
        ``mdata.uns``).

        :param name: View name.
        :returns: ``True`` when *name* can be read from this source.
        """
        if self.side == "sample" and name == TISSUE_IDENTIFIER:
            return TISSUE_IDENTIFIER in self._dataset.sample_meta.columns
        if self.side == "drug" and name == SMILES_COLUMN:
            return SMILES_COLUMN in self._dataset.response.var.columns
        if self.side == "drug" and name == DRUG_GRAPH_VIEW:
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
            if self.side == "drug":
                available = sorted(self._dataset.response.varm or ())
            else:
                available = sorted(public_view_name(key) for key in self._dataset.mdata.mod)
            raise KeyError(f"{self.side.capitalize()} view '{name}' not found. Available: {available}")
        index, data = found
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
        if self.side == "drug":
            data = (self._dataset.response.varm or {}).get(name)
            if data is None:
                return None
            if hasattr(data, "columns"):
                return tuple(data.columns.astype(str))
            return tuple(str(i) for i in range(data.shape[1]))
        stored = find_view(name, self._dataset.mdata.mod)
        return None if stored is None else tuple(self._dataset.mdata.mod[stored].var_names)

    def view_settings(self, name: str) -> dict[str, Any] | None:
        """Return the settings view *name* was precomputed with, or ``None`` when none are recorded.

        :param name: View name.
        :returns: The recorded settings; ``None`` for a view the dataset ships without a record.
        """
        return precomputed_views(self._dataset.mdata).get(self.side, {}).get(name)

    def expression_unit(self, name: str) -> str | None:
        """Return the ``expression_unit`` recorded in the ``uns`` of sample view *name*.

        :param name: View name.
        :returns: The unit, or ``None`` when the view is absent or records none.
        """
        stored = find_view(name, self._dataset.mdata.mod)
        if stored is None:
            return None
        unit = self._dataset.mdata.mod[stored].uns.get(EXPRESSION_UNIT_KEY)
        return None if unit is None else str(unit)

    def smiles(self, ids: np.ndarray) -> pd.Series | None:
        """Return the canonical SMILES of the drugs *ids*.

        :param ids: Drug IDs.
        :returns: SMILES indexed by *ids* (NaN for unknown drugs), or ``None`` when the dataset has none.
        """
        var = self._dataset.response.var
        if SMILES_COLUMN not in var.columns:
            return None
        return var[SMILES_COLUMN].reindex(ids)

    def graphs(self, ids: np.ndarray) -> list[dict[str, np.ndarray] | None]:
        """Return the stored graph of every drug in *ids*.

        Each graph dict has the keys ``"x"``, ``"edge_index"`` and ``"edge_attr"``.

        :param ids: Drug IDs.
        :returns: One graph per ID, ``None`` for a drug without one (every drug when none are stored).
        """
        graphs = self._dataset.mdata.uns.get(DRUG_GRAPHS_KEY, {})
        return [graphs.get(drug_id) for drug_id in np.asarray(ids, dtype=str)]

    def tissue(self, ids: np.ndarray) -> np.ndarray | None:
        """Return the tissue label of every sample in *ids*.

        :param ids: Sample IDs.
        :returns: One label per ID (NaN for unknown samples), or ``None`` when the dataset has no tissues.
        """
        if TISSUE_IDENTIFIER not in self._dataset.sample_meta.columns:
            return None
        return self._dataset.get_tissue(ids)

    def uns(self, key: str) -> Any:
        """Return the dataset-level auxiliary value stored under *key* in ``mdata.uns``.

        :param key: ``mdata.uns`` key, such as ``"pathways_gmt"`` or ``"sparsego"``.
        :returns: The stored value, or ``None`` when the dataset has none.
        """
        return self._dataset.mdata.uns.get(key)
