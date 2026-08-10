"""MuData-backed dataset class for drevalpy.

``Dataset`` wraps a MuData object and provides typed access to response data,
cell-line and drug features, metadata, and auxiliary model data. It replaces both
legacy response arrays and feature dicts with a single entry point backed by an
.h5mu file.
"""

from __future__ import annotations

from typing import Any

import mudata as md
import numpy as np
import pandas as pd
from upath import UPath as Path

from drevalpy.log import get_logger

from .dataset_utils.feature_access import (
    _get_obsm_features,
    _resolve_varm_key,
)
from .dataset_utils.feature_access import (
    available_drug_views as _available_drug_views,
)
from .dataset_utils.feature_access import (
    get_cell_line_feature_names as _get_cl_feature_names,
)
from .dataset_utils.feature_access import (
    get_cell_line_features as _get_cl_features,
)
from .dataset_utils.feature_access import (
    get_drug_feature_names as _get_drug_feature_names,
)
from .dataset_utils.feature_access import (
    get_drug_features as _get_drug_features,
)
from .dataset_utils.feature_access import (
    get_drug_graphs as _get_drug_graphs,
)
from .dataset_utils.randomization import with_randomized_views as _with_randomized_views
from .dataset_utils.sampling import _sample_hp_configs
from .mudatalike import MuDataLike

logger = get_logger(__name__)


class Dataset(MuDataLike):
    """Single entry point for all dataset access in drevalpy.

    Wraps a MuData object containing a "response" modality (cell_line x drug
    matrix with LN_IC50 as X) plus any number of cell-line feature modalities
    (gene_expression, proteomics, etc.).

    Drug features are stored as ``response.varm`` entries, drug graphs in
    ``mdata.uns["drug_graphs"]``, and model-specific auxiliary data in other
    ``mdata.uns`` keys.
    """

    def __init__(
        self,
        mdata: md.MuData,
        *,
        name: str,
        randomization: tuple[str, str] | None = None,
    ) -> None:
        """Wrap an existing MuData object.

        Args:
            mdata: A MuData object with at least a "response" modality.
            name: Human-readable dataset name.
            randomization: Optional (mode, view) tuple describing which
                randomization was applied.

        Raises:
            KeyError: If the "response" modality is missing.
        """
        if "response" not in mdata.mod:
            raise KeyError("MuData must contain a 'response' modality.")
        self._mdata = mdata
        self._name = name
        self.randomization = randomization

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        """Read a Dataset from an .h5mu file on disk.

        :param path: Path to the .h5mu file.
        :returns: A Dataset wrapping the loaded MuData.
        """
        from upath import UPath as Path

        resolved = Path(path)
        md.set_options(pull_on_update=False)
        mdata = md.read_h5mu(resolved)

        stored_name = mdata.uns.get("dataset_name")
        name = stored_name if isinstance(stored_name, str) else resolved.stem

        randomization = None
        stored_rand = mdata.uns.get("randomization")
        if isinstance(stored_rand, (list, tuple)) and len(stored_rand) == 2:
            randomization = (str(stored_rand[0]), str(stored_rand[1]))

        return cls(mdata, name=name, randomization=randomization)

    def save(self, path: str | Path) -> None:
        """Write this Dataset to an .h5mu file, preserving name and randomization.

        :param path: Output file path.
        """
        from upath import UPath as Path

        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)

        self._mdata.uns["dataset_name"] = self._name
        if self.randomization is not None:
            self._mdata.uns["randomization"] = list(self.randomization)
        elif "randomization" in self._mdata.uns:
            del self._mdata.uns["randomization"]

        self._mdata.write(str(resolved))

    def precompute(
        self,
        featurizer_cls: type,
        hyperparameters: list[dict] | int,
    ) -> None:
        """Pre-compute and store featurizer representations for given HP configs.

        :param featurizer_cls: Registered featurizer class (knows its own side).
        :param hyperparameters: Either a list of explicit HP dicts,
            or an int N to sample N configs from the featurizer's HP space.
        """
        from drevalpy.components.core.features.feature_source import CellLineFeatureSource, DrugFeatureSource

        if isinstance(hyperparameters, int):
            configs = _sample_hp_configs(featurizer_cls, hyperparameters)
        else:
            configs = list(hyperparameters)

        side = getattr(featurizer_cls, "side", "cell_line")
        if side == "cell_line":
            entity_ids = self.cell_line_ids
            source = CellLineFeatureSource(self, entity_ids)
        else:
            entity_ids = self.drug_ids
            source = DrugFeatureSource(self, entity_ids)

        from rich.progress import Progress

        with Progress() as progress:
            task = progress.add_task(f"Precomputing {featurizer_cls.storage_key}", total=len(configs))
            for config in configs:
                featurizer = featurizer_cls(**config) if config else featurizer_cls()
                featurizer.fit(source, entity_ids=entity_ids)
                matrix = featurizer.transform(source, entity_ids)
                featurizer.store(self._mdata, entity_ids, matrix, hyperparameters=config)
                progress.advance(task)

    @property
    def name(self) -> str:
        """Human-readable dataset name."""
        return self._name

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
        if modality == "pathway_features":
            if "pathway_features" not in self.response.obsm:
                return frozenset()
            data = np.asarray(self.response.obsm["pathway_features"])
            valid = ~np.all(np.isnan(data), axis=1)
            return frozenset(np.asarray(self.response.obs_names)[valid])

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

        varm_key = _resolve_varm_key(self._mdata, name)
        if varm_key is None:
            raise KeyError(f"Drug feature '{name}' not found. Available varm keys: {self.available_drug_views}")

        varm_data = np.asarray(self.response.varm[varm_key])
        valid = ~np.all(np.isnan(varm_data), axis=1)
        return frozenset(np.asarray(self.response.var_names)[valid])

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

        Args:
            modality: Name of the modality (e.g. "gene_expression").
            ids: 1-D array of cell line IDs to retrieve.
            strict: If True, raise KeyError for missing IDs instead of warning.

        Returns:
            Float32 array of shape (len(ids), n_features), rows aligned to *ids*.

        Raises:
            KeyError: If the modality is not present, or if *strict* and IDs are missing.
        """
        return _get_cl_features(self._mdata, modality, ids, strict=strict)

    def _get_obsm_features(self, key: str, ids: np.ndarray, *, strict: bool = False) -> np.ndarray:
        """Retrieve cell-line features stored in response.obsm."""
        return _get_obsm_features(self._mdata, key, ids, strict=strict)

    def get_cell_line_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return the feature (column) names for a cell-line view.

        Args:
            view: Name of the modality.

        Returns:
            Tuple of feature names, or None if names are unavailable.
        """
        return _get_cl_feature_names(self._mdata, view)

    # ------------------------------------------------------------------
    # Drug features
    # ------------------------------------------------------------------

    @property
    def available_drug_views(self) -> list[str]:
        """Sorted list of drug feature varm keys."""
        return _available_drug_views(self._mdata)

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
        return _get_drug_features(self._mdata, name, ids, strict=strict)

    def get_drug_feature_names(self, view: str) -> tuple[str, ...] | None:
        """Return the feature (column) names for a drug view stored in response.varm.

        Args:
            view: Drug view name (e.g. "chemberta", "morgan_fingerprint").

        Returns:
            Tuple of column name strings, or None if the view does not exist.
        """
        return _get_drug_feature_names(self._mdata, view)

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
        return _get_drug_graphs(self._mdata, ids)

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

    def subset_cell_lines(self, ids: np.ndarray) -> Dataset:
        """Return a new Dataset restricted to the given cell lines.

        Only keeps cell lines present in the response modality. Other modalities
        are also subset to their intersection with *ids*.

        Args:
            ids: 1-D array of cellosaurus IDs to keep.

        Returns:
            New Dataset backed by a view of the underlying MuData.
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
        return Dataset(new_mdata, name=self._name)

    def subset_drugs(self, ids: np.ndarray) -> Dataset:
        """Return a new Dataset restricted to the given drugs.

        Only the response modality has a drug axis; it is subset on var.
        Other modalities (cell-line features) are kept unchanged.

        Args:
            ids: 1-D array of PubChem drug IDs to keep.

        Returns:
            New Dataset backed by a view of the underlying MuData.
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
        return Dataset(new_mdata, name=self._name)

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
    # Randomization
    # ------------------------------------------------------------------

    def with_randomized_views(
        self,
        views: list[str],
        randomization_type: str = "permutation",
        random_state: int | None = None,
        *,
        randomization: tuple[str, str] | None = None,
    ) -> Dataset:
        """Return a copy of this Dataset with specified views randomized.

        For cell-line views (modalities or obsm keys), rows are permuted across
        cell lines. For drug views (varm keys), rows are permuted across drugs.
        For uns dict keys, values are reassigned to shuffled keys.

        Args:
            views: View names to randomize.
            randomization_type: "permutation" shuffles rows; "invariant" replaces
                each row with a random sample matching its mean and std.
            random_state: Seed for reproducibility.
            randomization: Optional (mode, view) tuple to attach to the new dataset.

        Returns:
            A new Dataset with the specified views randomized.

        Raises:
            ValueError: If randomization_type is not recognized.
            KeyError: If a view is not found in any storage location.
        """
        return _with_randomized_views(
            self,
            views,
            randomization_type=randomization_type,
            random_state=random_state,
            randomization=randomization,
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a formatted summary."""
        n_cl = len(self.cell_line_ids)
        n_dr = len(self.drug_ids)
        response = self.response_matrix
        n_measured = int(np.sum(~np.isnan(response)))
        mods = [m for m in self._mdata.mod.keys() if m != "response"]

        lines = [
            "Dataset",
            f"    Name: {self._name}",
            f"    Cell lines: {n_cl}",
            f"    Drugs: {n_dr}",
            f"    Measured pairs: {n_measured}",
            f"    Randomization: {self.randomization[0]} ({self.randomization[1]})"
            if self.randomization
            else "    Randomization: None",
            "    Modalities:",
        ]
        for mod in mods:
            shape = self._mdata.mod[mod].X.shape
            lines.append(f"        {mod}: {shape[0]} × {shape[1]}")

        return "\n".join(lines)
