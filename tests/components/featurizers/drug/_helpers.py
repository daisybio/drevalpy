"""In-memory drug feature source for the drug featurizer tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from drevalpy.types.data.feature_source import FeatureSource
from drevalpy.types.data.views import DRUG_GRAPH_VIEW


class DrugSource(FeatureSource):
    """In-memory drug feature source: per-drug view values, optional SMILES and feature names."""

    def __init__(
        self,
        views: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        smiles: Mapping[str, Any] | None = None,
        names: Mapping[str, tuple[str, ...]] | None = None,
        settings: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Serve *views* (``{view: {drug: value}}``), *smiles* (``{drug: smiles}``), *names* and view *settings*."""
        self.side = "drug"
        self._views = {view: dict(rows) for view, rows in (views or {}).items()}
        self._smiles = None if smiles is None else dict(smiles)
        self._names = dict(names or {})
        self._settings = {view: dict(values) for view, values in (settings or {}).items()}
        drugs = dict.fromkeys(drug for rows in self._views.values() for drug in rows)
        drugs.update(dict.fromkeys(self._smiles or {}))
        self._ids = np.array(list(drugs), dtype=str)

    @property
    def identifiers(self) -> np.ndarray:
        """Every drug the mock serves, in first-seen order."""
        return self._ids

    def has_view(self, name: str) -> bool:
        """Return whether *name* is served."""
        return name in self._views

    def view_settings(self, name: str) -> dict[str, Any] | None:
        """Return the settings recorded for *name*, ``None`` when unrecorded."""
        return self._settings.get(name)

    def view(self, name: str, ids: np.ndarray) -> np.ndarray:
        """Stack the rows of a dense view; ``KeyError`` for an absent view or drug.

        A view of per-drug matrices (MolGNet node features) comes back as an object array instead, ``None``
        for a drug the view lacks.
        """
        rows = self._views[name]
        if any(np.ndim(value) > 1 for value in rows.values()):
            out = np.empty(len(ids), dtype=object)
            for index, drug in enumerate(ids):
                out[index] = rows.get(str(drug))
            return out
        return np.vstack([np.asarray(rows[str(drug)], dtype=np.float64).ravel() for drug in ids])

    def feature_names(self, name: str) -> tuple[str, ...] | None:
        """Return the names given for *name*."""
        return self._names.get(name)

    def smiles(self, ids: np.ndarray) -> pd.Series | None:
        """Return the SMILES of *ids* (NaN for unknown drugs), or ``None`` without SMILES."""
        if self._smiles is None:
            return None
        return pd.Series(self._smiles, dtype=object).reindex(ids)

    def graphs(self, ids: np.ndarray) -> list[Any]:
        """Return the stored ``drug_graph`` value of every drug, ``None`` where it has none."""
        stored = self._views.get(DRUG_GRAPH_VIEW, {})
        return [stored.get(str(drug)) for drug in ids]
