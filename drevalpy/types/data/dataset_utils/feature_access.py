"""Mixin providing cell-line and drug feature access from MuData."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .aligned_fetch import _aligned_fetch

if TYPE_CHECKING:
    import mudata as md


class FeatureAccessMixin:
    """Mixin that provides cell-line and drug feature access methods.

    Expects ``self._mdata`` to be a MuData object with a "response" modality.
    """

    _mdata: md.MuData

    def get_cell_line_features(self, modality: str, ids: np.ndarray, *, strict: bool = False) -> np.ndarray:
        """Get a feature matrix for the specified cell lines from a modality.

        Args:
            modality: Name of the modality (e.g. "gene_expression").
            ids: 1-D array of cell line IDs to retrieve.
            strict: If True, raise KeyError for missing IDs instead of warning.

        Returns:
            Float32 array of shape (len(ids), n_features), rows aligned to *ids*.

        Raises:
            KeyError: If the modality is not present, or if *strict* and IDs are missing.
        """
        ids = np.asarray(ids, dtype=str)

        if modality == "pathway_features":
            return self._get_obsm_features("pathway_features", ids, strict=strict)

        if modality not in self._mdata.mod:
            raise KeyError(f"Modality '{modality}' not found. Available: {list(self._mdata.mod.keys())}")

        adata = self._mdata.mod[modality]
        x = adata.X
        if hasattr(x, "toarray"):
            x = x.toarray()
        return _aligned_fetch(pd.Index(adata.obs_names), ids, np.asarray(x), strict=strict, entity_label="cell line")

    def _get_obsm_features(self, key: str, ids: np.ndarray, *, strict: bool = False) -> np.ndarray:
        """Retrieve cell-line features stored in response.obsm."""
        response = self._mdata.mod["response"]
        if key not in response.obsm:
            raise KeyError(f"obsm key '{key}' not found in response modality.")

        obsm_data = np.asarray(response.obsm[key])
        return _aligned_fetch(pd.Index(response.obs_names), ids, obsm_data, strict=strict, entity_label="cell line")

    def get_cell_line_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return the feature (column) names for a cell-line view.

        Args:
            view: Name of the modality.

        Returns:
            Tuple of feature names, or None if names are unavailable.
        """
        if view == "pathway_features":
            return None
        if view not in self._mdata.mod:
            return None
        return tuple(self._mdata.mod[view].var_names)

    def _resolve_varm_key(self, name: str) -> str | None:
        """Resolve a varm key by exact match or prefix match (name:variant)."""
        varm = self._mdata.mod["response"].varm
        if varm is None:
            return None
        if name in varm:
            return name
        for key in varm.keys():
            if key.startswith(name + ":"):
                return key
        return None

    @property
    def available_drug_views(self) -> list[str]:
        """Sorted list of drug feature varm keys."""
        response = self._mdata.mod["response"]
        if response.varm is None:
            return []
        return sorted(response.varm.keys())

    def get_drug_features(self, name: str, ids: np.ndarray, *, strict: bool = False) -> np.ndarray:
        """Get a drug feature matrix from response.varm, aligned to given IDs.

        Args:
            name: Key in ``response.varm`` (e.g. "chemberta", "morgan_fingerprint").
            ids: 1-D array of drug (PubChem) IDs.
            strict: If True, raise KeyError for missing IDs instead of warning.

        Returns:
            Float32 array of shape (len(ids), n_features), rows aligned to *ids*.

        Raises:
            KeyError: If the varm key does not exist, or if *strict* and IDs are missing.
        """
        varm_key = self._resolve_varm_key(name)
        if varm_key is None:
            raise KeyError(f"Drug feature '{name}' not found. Available varm keys: {self.available_drug_views}")

        response = self._mdata.mod["response"]
        ids = np.asarray(ids, dtype=str)
        varm_data = np.asarray(response.varm[varm_key])
        return _aligned_fetch(pd.Index(response.var_names), ids, varm_data, strict=strict, entity_label="drug")

    def get_drug_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return the feature (column) names for a drug view stored in response.varm.

        Args:
            view: Drug view name (e.g. "chemberta", "morgan_fingerprint").

        Returns:
            Tuple of column name strings, or None if the view does not exist.
        """
        varm_key = self._resolve_varm_key(view)
        if varm_key is None:
            return None
        varm_data = self._mdata.mod["response"].varm[varm_key]
        if hasattr(varm_data, "columns"):
            return tuple(varm_data.columns.astype(str))
        return tuple(str(i) for i in range(varm_data.shape[1]))

    def get_drug_graphs(self, ids: np.ndarray) -> list[dict[str, np.ndarray] | None]:
        """Get PyTorch Geometric graph data for the specified drugs.

        Each graph dict has keys "x", "edge_index", "edge_attr" with numpy arrays.
        Returns None for drugs without a stored graph.

        Args:
            ids: 1-D array of drug (PubChem) IDs.

        Returns:
            List of graph dicts (or None) aligned to *ids*.

        Raises:
            KeyError: If "drug_graphs" is not in mdata.uns.
        """
        if "drug_graphs" not in self._mdata.uns:
            raise KeyError("'drug_graphs' not found in mdata.uns.")

        ids = np.asarray(ids, dtype=str)
        graphs = self._mdata.uns["drug_graphs"]
        return [graphs.get(drug_id) for drug_id in ids]

    def entities_with_modality(self, modality: str, *, side: str = "cell_line") -> frozenset[str]:
        """Return entity IDs that have actual feature data for a modality.

        Args:
            modality: Modality or view name (e.g. "gene_expression", "fingerprints").
            side: Either "cell_line" or "drug".

        Returns:
            Frozenset of entity IDs that have non-NaN data for the modality.

        Raises:
            KeyError: If the modality/view is not found.
        """
        if side == "cell_line":
            return self._cell_line_entities_for_modality(modality)
        return self._drug_entities_for_view(modality)

    def _cell_line_entities_for_modality(self, modality: str) -> frozenset[str]:
        """Cell line IDs present in a given modality."""
        response = self._mdata.mod["response"]
        if modality == "pathway_features":
            if "pathway_features" not in response.obsm:
                return frozenset()
            data = np.asarray(response.obsm["pathway_features"])
            valid = ~np.all(np.isnan(data), axis=1)
            return frozenset(np.asarray(response.obs_names)[valid])

        if modality not in self._mdata.mod:
            raise KeyError(f"Modality '{modality}' not found. Available: {list(self._mdata.mod.keys())}")

        adata = self._mdata.mod[modality]
        x = adata.X
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = np.asarray(x)
        valid = ~np.all(np.isnan(x), axis=1)
        return frozenset(np.asarray(adata.obs_names)[valid])

    def _drug_entities_for_view(self, name: str) -> frozenset[str]:
        """Drug IDs present in a given drug feature view."""
        if name == "drug_graph":
            if "drug_graphs" not in self._mdata.uns:
                return frozenset()
            return frozenset(str(k) for k in self._mdata.uns["drug_graphs"].keys())

        varm_key = self._resolve_varm_key(name)
        if varm_key is None:
            raise KeyError(f"Drug feature '{name}' not found. Available varm keys: {self.available_drug_views}")

        response = self._mdata.mod["response"]
        varm_data = np.asarray(response.varm[varm_key])
        valid = ~np.all(np.isnan(varm_data), axis=1)
        return frozenset(np.asarray(response.var_names)[valid])
