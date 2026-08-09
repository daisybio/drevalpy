"""MuData-backed dataset class for drevalpy.

``MuDataset`` wraps a MuData object and provides typed access to response data,
cell-line and drug features, metadata, and auxiliary model data. It replaces both
legacy response arrays and feature dicts with a single entry point backed by an
.h5mu file.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from upath import UPath as Path

import mudata as md
from drevalpy.log import get_logger

from .mudatalike import MuDataLike

logger = get_logger(__name__)


def _aligned_fetch(
    index: pd.Index,
    ids: np.ndarray,
    data: np.ndarray,
    *,
    strict: bool,
    entity_label: str,
) -> np.ndarray:
    """Fetch rows from *data* aligned to *ids* using *index*, filling NaN for missing.

    Args:
        index: pd.Index mapping entity names to row positions in *data*.
        ids: 1-D array of requested entity IDs.
        data: 2-D source array to fetch rows from.
        strict: If True, raise KeyError for missing IDs instead of warning.
        entity_label: Human-readable label for error messages (e.g. "cell line").

    Returns:
        Float32 array of shape (len(ids), data.shape[1]).
    """
    positions = index.get_indexer(ids)
    missing_mask = positions == -1
    if missing_mask.any():
        n_missing = int(missing_mask.sum())
        sample = ids[missing_mask][:5].tolist()
        msg = f"{n_missing} of {len(ids)} {entity_label} IDs not found (first few: {sample}). Returning NaN rows."
        if strict:
            raise KeyError(msg)
        logger.warning(msg)

    n_features = data.shape[1]
    result = np.full((len(ids), n_features), np.nan, dtype=np.float32)
    valid = positions >= 0
    result[valid] = np.asarray(data[positions[valid]], dtype=np.float32)
    return result


def _randomize_matrix(data: np.ndarray, rng: np.random.Generator, randomization_type: str) -> np.ndarray:
    """Apply randomization to a 2-D feature matrix."""
    if randomization_type == "permutation":
        perm = rng.permutation(data.shape[0])
        return data[perm]
    return np.array(
        [rng.normal(row.mean(), max(row.std(), 1e-8), row.shape) for row in data],
        dtype=np.float32,
    )


def _randomize_single_view(
    view: str,
    modality_keys: set[str],
    varm_keys: set[str],
    new_mods: dict[str, md.AnnData],
    source: Any,
    rng: np.random.Generator,
    randomization_type: str,
) -> None:
    """Randomize a single view in-place within new_mods."""
    import anndata

    if view in modality_keys:
        adata = new_mods[view]
        x = adata.X
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = _randomize_matrix(np.asarray(x, dtype=np.float32), rng, randomization_type)
        new_mods[view] = anndata.AnnData(X=x, obs=adata.obs.copy(), var=adata.var.copy())
    elif view in varm_keys:
        resp = new_mods["response"]
        varm_data = np.asarray(resp.varm[view], dtype=np.float32)
        resp.varm[view] = _randomize_matrix(varm_data, rng, randomization_type)
    elif view == "pathway_features" and "pathway_features" in (source.response.obsm or {}):
        resp = new_mods["response"]
        obsm_data = np.asarray(resp.obsm["pathway_features"], dtype=np.float32)
        resp.obsm["pathway_features"] = _randomize_matrix(obsm_data, rng, randomization_type)
    else:
        raise ValueError(
            f"View {view!r} not found in modalities {sorted(modality_keys)}, "
            f"varm keys {sorted(varm_keys)}, or obsm keys."
        )


class MuDataset(MuDataLike):
    """Single entry point for all dataset access in drevalpy.

    Wraps a MuData object containing a "response" modality (cell_line x drug
    matrix with LN_IC50 as X) plus any number of cell-line feature modalities
    (gene_expression, proteomics, etc.).

    Drug features are stored as ``response.varm`` entries, drug graphs in
    ``mdata.uns["drug_graphs"]``, and model-specific auxiliary data in other
    ``mdata.uns`` keys.
    """

    _DRUG_VIEW_ALIASES: dict[str, str] = {
        "fingerprints": "morgan_fingerprint",
    }

    def __init__(self, mdata: md.MuData) -> None:
        """Wrap an existing MuData object.

        Args:
            mdata: A MuData object with at least a "response" modality.

        Raises:
            KeyError: If the "response" modality is missing.
        """
        if "response" not in mdata.mod:
            raise KeyError("MuData must contain a 'response' modality.")
        self._mdata = mdata
        self._drug_view_map: dict[str, str] = self._build_drug_view_map()

    @classmethod
    def from_file(cls, path: str | Path) -> MuDataset:
        """Read a MuDataset from an .h5mu file on disk.

        Args:
            path: Path to the .h5mu file.

        Returns:
            A MuDataset wrapping the loaded MuData.
        """
        md.set_options(pull_on_update=False)
        mdata = md.read_h5mu(path)
        return cls(mdata)

    @property
    def mdata(self) -> md.MuData:
        """Return the underlying MuData object."""
        return self._mdata

    # ------------------------------------------------------------------
    # Response access
    # ------------------------------------------------------------------

    @property
    def response(self) -> md.AnnData:
        """Return the response AnnData (cell_lines x drugs)."""
        return self._mdata.mod["response"]

    @property
    def response_matrix(self) -> np.ndarray:
        """LN_IC50 response matrix (n_cell_lines x n_drugs).

        Returns:
            Dense float32 array of shape (n_cell_lines, n_drugs).
        """
        x = self.response.X
        if hasattr(x, "toarray"):
            return np.asarray(x.toarray(), dtype=np.float32)
        return np.asarray(x, dtype=np.float32)

    @property
    def cell_line_ids(self) -> np.ndarray:
        """Cell line identifiers (obs_names of the response modality).

        Returns:
            1-D string array of cellosaurus IDs.
        """
        return np.asarray(self.response.obs_names)

    @property
    def drug_ids(self) -> np.ndarray:
        """Drug identifiers (var_names of the response modality).

        Returns:
            1-D string array of PubChem IDs.
        """
        return np.asarray(self.response.var_names)

    def get_response_layer(self, name: str) -> np.ndarray:
        """Retrieve a named response layer (e.g. "AUC").

        Args:
            name: Layer name within the response AnnData.

        Returns:
            Dense float32 array of shape (n_cell_lines, n_drugs).

        Raises:
            KeyError: If the layer does not exist.
        """
        if name not in self.response.layers:
            raise KeyError(f"Response layer '{name}' not found. Available: {list(self.response.layers.keys())}")
        layer = self.response.layers[name]
        if hasattr(layer, "toarray"):
            return np.asarray(layer.toarray(), dtype=np.float32)
        return np.asarray(layer, dtype=np.float32)

    # ------------------------------------------------------------------
    # Cell-line features
    # ------------------------------------------------------------------

    def get_cell_line_features(self, modality: str, ids: np.ndarray, *, strict: bool = False) -> np.ndarray:
        """Get a feature matrix for the specified cell lines from a modality.

        For standard omics modalities (gene_expression, proteomics, etc.) the
        features come from ``mdata.mod[modality].X``. For "pathway_features" the
        data comes from ``response.obsm["pathway_features"]``.

        Args:
            modality: Name of the modality or obsm key.
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
        if key not in self.response.obsm:
            raise KeyError(f"obsm key '{key}' not found in response modality.")

        obsm_data = np.asarray(self.response.obsm[key])
        return _aligned_fetch(
            pd.Index(self.response.obs_names), ids, obsm_data, strict=strict, entity_label="cell line"
        )

    def get_cell_line_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return the feature (column) names for a cell-line view.

        For standard modalities the names come from ``mdata.mod[view].var_names``.
        For "pathway_features" stored in ``response.obsm``, returns None (no
        named columns available).

        Args:
            view: Name of the modality or obsm key.

        Returns:
            Tuple of feature names, or None if names are unavailable.
        """
        if view == "pathway_features":
            return None
        if view not in self._mdata.mod:
            return None
        return tuple(self._mdata.mod[view].var_names)

    # ------------------------------------------------------------------
    # Drug features
    # ------------------------------------------------------------------

    def _build_drug_view_map(self) -> dict[str, str]:
        """Build a mapping from canonical/alias names to actual varm keys."""
        view_map: dict[str, str] = {}
        varm_keys = list(self.response.varm.keys()) if self.response.varm else []

        for key in varm_keys:
            view_map[key] = key
            if ":" in key:
                prefix = key.split(":", 1)[0]
                if prefix not in view_map:
                    view_map[prefix] = key
                else:
                    logger.warning(
                        "Drug view prefix %r already maps to %r; ignoring %r. Use the full key to access it.",
                        prefix,
                        view_map[prefix],
                        key,
                    )

        for alias, target in self._DRUG_VIEW_ALIASES.items():
            if target in view_map and alias not in view_map:
                view_map[alias] = view_map[target]

        return view_map

    @property
    def available_drug_views(self) -> list[str]:
        """Sorted list of canonical drug view names (keys of the view registry)."""
        return sorted(self._drug_view_map.keys())

    def _resolve_drug_view(self, name: str) -> str | None:
        """Resolve a drug view name to an actual varm key via the registry."""
        return self._drug_view_map.get(name)

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
        resolved = self._resolve_drug_view(name)
        if resolved is None:
            raise KeyError(f"Drug feature '{name}' not found. Available views: {self.available_drug_views}")

        ids = np.asarray(ids, dtype=str)
        varm_data = np.asarray(self.response.varm[resolved])
        return _aligned_fetch(pd.Index(self.response.var_names), ids, varm_data, strict=strict, entity_label="drug")

    def get_drug_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return the feature (column) names for a drug view stored in response.varm.

        Column names are read from ``response.varm`` using a DataFrame-backed
        varm entry or positional indices when no explicit names exist.

        Args:
            view: Drug view name (e.g. "chemberta", "morgan_fingerprint").

        Returns:
            Tuple of column name strings, or None if the view does not exist.
        """
        resolved = self._resolve_drug_view(view)
        if resolved is None:
            return None
        varm_data = self.response.varm[resolved]
        if hasattr(varm_data, "columns"):
            return tuple(varm_data.columns.astype(str))
        return tuple(str(i) for i in range(varm_data.shape[1]))

    # ------------------------------------------------------------------
    # Drug graphs
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def cell_line_meta(self) -> pd.DataFrame:
        """Global cell-line metadata (cell_line_name, tissue, etc.).

        Returns:
            DataFrame indexed by cellosaurus_id from mdata.obs.
        """
        return self._mdata.obs

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        """Get tissue labels for the given cell line IDs.

        Args:
            ids: 1-D array of cellosaurus IDs.

        Returns:
            1-D string array of tissue labels (NaN for unknown IDs).
        """
        ids = np.asarray(ids, dtype=str)
        idx = pd.Index(self._mdata.obs.index)
        positions = idx.get_indexer(ids)

        tissues = self._mdata.obs["tissue"].values
        result = np.full(len(ids), np.nan, dtype=object)
        valid = positions >= 0
        result[valid] = tissues[positions[valid]]
        return result

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------

    def subset_cell_lines(self, ids: np.ndarray) -> MuDataset:
        """Return a new MuDataset restricted to the given cell lines.

        Only keeps cell lines present in the response modality. Other modalities
        are also subset to their intersection with *ids*.

        Args:
            ids: 1-D array of cellosaurus IDs to keep.

        Returns:
            New MuDataset backed by a view of the underlying MuData.
        """
        ids = np.asarray(ids, dtype=str)
        response_mask = np.isin(self.response.obs_names, ids)
        kept_cell_lines = self.response.obs_names[response_mask]

        new_mods: dict[str, md.AnnData] = {}
        for mod_name, mod_adata in self._mdata.mod.items():
            mod_mask = np.isin(mod_adata.obs_names, kept_cell_lines)
            new_mods[mod_name] = mod_adata[mod_mask].copy()

        md.set_options(pull_on_update=False)
        new_mdata = md.MuData(new_mods)
        new_mdata.obs = self._mdata.obs.loc[self._mdata.obs.index.isin(kept_cell_lines)].copy()
        for key, val in self._mdata.uns.items():
            new_mdata.uns[key] = val
        return MuDataset(new_mdata)

    def subset_drugs(self, ids: np.ndarray) -> MuDataset:
        """Return a new MuDataset restricted to the given drugs.

        Only the response modality has a drug axis; it is subset on var.
        Other modalities (cell-line features) are kept unchanged.

        Args:
            ids: 1-D array of PubChem drug IDs to keep.

        Returns:
            New MuDataset backed by a view of the underlying MuData.
        """
        ids = np.asarray(ids, dtype=str)
        drug_mask = np.isin(self.response.var_names, ids)

        new_mods: dict[str, md.AnnData] = {}
        for mod_name, mod_adata in self._mdata.mod.items():
            if mod_name == "response":
                new_mods[mod_name] = mod_adata[:, drug_mask].copy()
            else:
                new_mods[mod_name] = mod_adata.copy()

        md.set_options(pull_on_update=False)
        new_mdata = md.MuData(new_mods)
        new_mdata.obs = self._mdata.obs.copy()
        for key, val in self._mdata.uns.items():
            new_mdata.uns[key] = val
        return MuDataset(new_mdata)

    # ------------------------------------------------------------------
    # Auxiliary data
    # ------------------------------------------------------------------

    def get_uns(self, key: str) -> Any:
        """Access arbitrary data from mdata.uns.

        Args:
            key: Key in the uns dict.

        Returns:
            The stored value.

        Raises:
            KeyError: If the key does not exist.
        """
        if key not in self._mdata.uns:
            raise KeyError(f"uns key '{key}' not found. Available: {list(self._mdata.uns.keys())}")
        return self._mdata.uns[key]

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Randomization
    # ------------------------------------------------------------------

    def with_randomized_views(
        self,
        views: list[str],
        randomization_type: str = "permutation",
        random_state: int | None = None,
    ) -> MuDataset:
        """Return a copy of this MuDataset with specified views randomized.

        For cell-line views (modalities or obsm keys), rows are permuted across
        cell lines. For drug views (varm keys), rows are permuted across drugs.

        Args:
            views: View names (modality names or varm keys) to randomize.
            randomization_type: "permutation" shuffles rows; "invariant" replaces
                each row with a random sample matching its mean and std.
            random_state: Seed for reproducibility.

        Returns:
            A new MuDataset with the specified views randomized.

        Raises:
            ValueError: If randomization_type is not recognized or a view is not found.
        """
        if randomization_type not in ("permutation", "invariant"):
            raise ValueError(f"Unknown randomization_type {randomization_type!r}. Use 'permutation' or 'invariant'.")

        import copy

        rng = np.random.default_rng(random_state)

        new_mods: dict[str, md.AnnData] = {}
        for mod_name, mod_adata in self._mdata.mod.items():
            new_mods[mod_name] = mod_adata.copy()

        varm_keys = set(self.response.varm.keys()) if self.response.varm else set()
        modality_keys = set(self._mdata.mod.keys()) - {"response"}

        for view in views:
            _randomize_single_view(view, modality_keys, varm_keys, new_mods, self, rng, randomization_type)

        md.set_options(pull_on_update=False)
        new_mdata = md.MuData(new_mods)
        new_mdata.obs = self._mdata.obs.copy()
        for key, val in self._mdata.uns.items():
            new_mdata.uns[key] = copy.deepcopy(val) if isinstance(val, dict) else val
        return MuDataset(new_mdata)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a summary string."""
        n_cl = len(self.cell_line_ids)
        n_dr = len(self.drug_ids)
        mods = list(self._mdata.mod.keys())
        return f"MuDataset(cell_lines={n_cl}, drugs={n_dr}, modalities={mods})"
