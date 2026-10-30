"""In-memory drug feature source for the drug featurizer tests."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from drevalpy.types.data.feature_source import FeatureSource


class DrugSource(FeatureSource):
    """In-memory drug feature source: per-drug view values, optional SMILES and feature names."""

    def __init__(
        self,
        views: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        smiles: Mapping[str, Any] | None = None,
        names: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Serve *views* (``{view: {drug: value}}``), *smiles* (``{drug: smiles}``) and *names*."""
        self.side = "drug"
        self._views = {view: dict(rows) for view, rows in (views or {}).items()}
        self._smiles = None if smiles is None else dict(smiles)
        self._names = dict(names or {})
        drugs = dict.fromkeys(drug for rows in self._views.values() for drug in rows)
        drugs.update(dict.fromkeys(self._smiles or {}))
        self._ids = np.array(list(drugs), dtype=str)

    @property
    def mdata(self) -> Any:
        """A stub MuData holding only the SMILES column, or ``None`` without SMILES."""
        if self._smiles is None:
            return None
        frame = pd.DataFrame({"canonical_smiles": pd.Series(self._smiles, dtype=object)})
        return SimpleNamespace(mod={"response": SimpleNamespace(var=frame)})

    def has_view(self, view: str) -> bool:
        """Return whether *view* is served."""
        return view in self._views

    def get_view_matrix(self, view: str, entity_ids: np.ndarray) -> np.ndarray:
        """Stack the rows of a dense view; ``KeyError`` for an absent view or drug."""
        rows = self._views[view]
        return np.vstack([np.asarray(rows[str(drug)], dtype=np.float64).ravel() for drug in entity_ids])

    def get_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return the names given for *view*."""
        return self._names.get(view)

    def get_entity_view(self, entity_id: str, view: str) -> Any:
        """Return one drug's value of *view*, ``None`` when absent."""
        return self._views.get(view, {}).get(str(entity_id))
