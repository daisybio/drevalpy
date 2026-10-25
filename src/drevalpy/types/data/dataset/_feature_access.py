"""Mixin providing sample and drug feature access from MuData."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy.errors import InvalidInputError
from drevalpy.types.data.views import find_view, public_view_name

from ._dense import to_dense
from .aligned_fetch import aligned_fetch

if TYPE_CHECKING:
    import mudata as md


class FeatureAccessMixin:
    """Mixin that provides sample and drug feature access methods.

    Expects ``self._mdata`` to be a MuData object with a "response" modality.

    Sample views are addressed by their public name (the name configs, recipes
    and predictors use). Every lookup of ``self._mdata.mod`` by a caller-supplied
    name goes through :func:`~drevalpy.types.data.views.find_view`.
    """

    _mdata: md.MuData

    def _modality(self, view: str) -> str | None:
        """Return the stored modality name backing *view*, or None if the MuData has none."""
        return find_view(view, self._mdata.mod)

    def get_sample_features(self, modality: str, ids: np.ndarray, *, strict: bool = False) -> np.ndarray:
        """Get a feature matrix for the specified samples from a modality.

        :param modality: Public name of the modality (e.g. "gene_expression").
        :param ids: 1-D array of sample IDs to retrieve.
        :param strict: If True, raise KeyError for missing IDs instead of warning.
        :returns: Float32 array of shape (len(ids), n_features), rows aligned to *ids*.
        :raises KeyError: If the modality is not present, or if *strict* and IDs are missing.
        """
        import pandas as pd

        ids = np.asarray(ids, dtype=str)

        if modality == "pathway_features":
            return self._get_obsm_features("pathway_features", ids, strict=strict)

        stored = self._modality(modality)
        if stored is None:
            available = sorted(public_view_name(key) for key in self._mdata.mod)
            raise KeyError(f"Modality '{modality}' not found. Available: {available}")
        adata = self._mdata.mod[stored]
        x = np.asarray(to_dense(adata.X))
        return aligned_fetch(pd.Index(adata.obs_names), ids, x, strict=strict, entity_label="sample")

    def _get_obsm_features(self, key: str, ids: np.ndarray, *, strict: bool = False) -> np.ndarray:
        """Retrieve sample features stored in response.obsm."""
        import pandas as pd

        response = self._mdata.mod["response"]
        if key not in response.obsm:
            raise KeyError(f"obsm key '{key}' not found in response modality.")

        obsm_data = np.asarray(response.obsm[key])
        return aligned_fetch(pd.Index(response.obs_names), ids, obsm_data, strict=strict, entity_label="sample")

    def get_sample_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return the feature (column) names for a sample view.

        :param view: Public name of the modality.
        :returns: Tuple of feature names, or None if names are unavailable.
        """
        if view == "pathway_features":
            return None
        stored = self._modality(view)
        if stored is None:
            return None
        return tuple(self._mdata.mod[stored].var_names)

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

    def get_drug_features(self, name: str, ids: np.ndarray, *, strict: bool = False) -> np.ndarray:
        """Get a drug feature matrix from response.varm, aligned to given IDs.

        :param name: Key in ``response.varm`` (e.g. "chemberta", "morgan_fingerprint").
        :param ids: 1-D array of drug (PubChem) IDs.
        :param strict: If True, raise KeyError for missing IDs instead of warning.
        :returns: Float32 array of shape (len(ids), n_features), rows aligned to *ids*.
        :raises KeyError: If the varm key does not exist, or if *strict* and IDs are missing.
        :raises InvalidInputError: If the view is an object array of per-drug arrays (such as ragged MolGNet node
            features), which has no matrix form.
        """
        import pandas as pd

        response = self._mdata.mod["response"]
        varm_key = self._resolve_varm_key(name)
        if varm_key is None:
            raise KeyError(f"Drug feature '{name}' not found. Available varm keys: {sorted(response.varm or ())}")

        ids = np.asarray(ids, dtype=str)
        varm_data = np.asarray(response.varm[varm_key])
        if varm_data.dtype == object:
            try:
                varm_data = varm_data.astype(np.float32)
            except (ValueError, TypeError) as err:
                msg = (
                    f"Drug view '{varm_key}' in response.varm is an object array (one array per drug, such as "
                    "ragged per-atom node features) and cannot be read as a (n_drugs, n_features) matrix. A .h5mu "
                    "file cannot store such a view; remove the key so the featurizer computes it from the SMILES "
                    "instead."
                )
                raise InvalidInputError(msg) from err
        return aligned_fetch(pd.Index(response.var_names), ids, varm_data, strict=strict, entity_label="drug")

    def get_drug_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return the feature (column) names for a drug view stored in response.varm.

        :param view: Drug view name (e.g. "chemberta", "morgan_fingerprint").
        :returns: Tuple of column name strings, or None if the view does not exist.
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

        :param ids: 1-D array of drug (PubChem) IDs.
        :returns: List of graph dicts (or None) aligned to *ids*.
        :raises KeyError: If "drug_graphs" is not in mdata.uns.
        """
        if "drug_graphs" not in self._mdata.uns:
            raise KeyError("'drug_graphs' not found in mdata.uns.")

        ids = np.asarray(ids, dtype=str)
        graphs = self._mdata.uns["drug_graphs"]
        return [graphs.get(drug_id) for drug_id in ids]
