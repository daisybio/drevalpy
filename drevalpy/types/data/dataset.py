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

from .dataset_utils.feature_access import FeatureAccessMixin
from .dataset_utils.randomization import RandomizationMixin
from .dataset_utils.sampling import _sample_hp_configs
from .mudatalike import MuDataLike

logger = get_logger(__name__)


class Dataset(FeatureAccessMixin, RandomizationMixin, MuDataLike):
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
        view: str | None = None,
    ) -> None:
        """Pre-compute and store featurizer representations for given HP configs.

        For independent featurizers (those with ``_compute_from_source``), calls
        that method directly, bypassing fit/transform. Always includes the
        default HP config in addition to sampled variants.

        :param featurizer_cls: Registered featurizer class (knows its own side).
        :param hyperparameters: Either a list of explicit HP dicts,
            or an int N to sample N configs from the featurizer's HP space.
        :param view: View name for view-parameterized featurizers (e.g., "gene_expression").
        """
        from drevalpy.components.core.features.feature_source import CellLineFeatureSource, DrugFeatureSource

        if isinstance(hyperparameters, int):
            configs = _sample_hp_configs(featurizer_cls, hyperparameters)
        else:
            configs = list(hyperparameters)

        default_config = featurizer_cls.get_default_hyperparameters()
        if default_config not in configs:
            configs.insert(0, default_config)

        side = getattr(featurizer_cls, "side", "cell_line")
        if side == "cell_line":
            entity_ids = self.cell_line_ids
            source = CellLineFeatureSource(self, entity_ids)
        else:
            entity_ids = self.drug_ids
            source = DrugFeatureSource(self, entity_ids)

        base_kwargs: dict = {}
        if view is not None:
            base_kwargs["view"] = view

        from rich.progress import Progress

        with Progress() as progress:
            task = progress.add_task(f"Precomputing {featurizer_cls.storage_key}", total=len(configs))
            for config in configs:
                featurizer = featurizer_cls(**{**base_kwargs, **config})
                if hasattr(featurizer, "_compute_from_source"):
                    matrix = featurizer._compute_from_source(source, entity_ids)
                else:
                    featurizer.fit(source, entity_ids=entity_ids)
                    matrix = featurizer.transform(source, entity_ids)
                featurizer.store(self._mdata, entity_ids, matrix, hyperparameters=config)
                progress.advance(task)

    def precompute_all(self, n_variants: int = 10) -> None:
        """Pre-compute all fixed featurizers with N HP variants each.

        Iterates all registered featurizers (cell-line + drug), skips those
        not marked for precomputation, and pre-computes N sampled HP
        configurations for the rest. Featurizers without any HP space get a
        single default-params variant.

        :param n_variants: Number of HP configurations to sample for featurizers
            that have a tunable HP space.
        """
        from drevalpy.components.registry.featurizer_registry import (
            cell_line_featurizer_registry,
            drug_featurizer_registry,
        )

        for registry in (cell_line_featurizer_registry, drug_featurizer_registry):
            for name in registry.list_names():
                self._precompute_single(registry.get(name), n_variants)

    def _precompute_single(self, cls: type, n_variants: int) -> None:
        """Pre-compute one featurizer class if eligible."""
        name = getattr(cls, "registry_name", cls.__name__)
        if not cls.precompute:
            return
        if cls.entity_id_only:
            logger.debug("Skipping %s: entity_id_only", name)
            return
        source_views = getattr(cls, "source_views", None)
        if source_views and not self._has_source_data(source_views, cls):
            logger.debug("Skipping %s: required source data not available", name)
            return
        hp_space = cls.get_hyperparameter_space()
        effective_n = n_variants if hp_space else 1
        view: str | None = None
        if cls.requires_view:
            if cls.input_views:
                view = cls.input_views[0]
            else:
                logger.debug("Skipping %s: requires_view but no input_views declared", name)
                return
        try:
            logger.info("Precomputing %s (%d variants)", name, effective_n)
            self.precompute(cls, effective_n, view=view)
        except (ValueError, TypeError, KeyError, ImportError) as exc:
            logger.warning("Failed to precompute %s: %s", name, exc)

    def _has_source_data(self, source_views: tuple[str, ...], cls: type) -> bool:
        """Check if the dataset has the raw source data needed for a featurizer."""
        side = getattr(cls, "side", "cell_line")
        return all(self._has_single_source(view, side) for view in source_views)

    def _has_single_source(self, view: str, side: str) -> bool:
        """Check availability of a single source view."""
        if view == "canonical_smiles":
            response = self._mdata.mod.get("response")
            return response is not None and "canonical_smiles" in response.var.columns
        if side == "cell_line":
            return view in self._mdata.mod
        response = self._mdata.mod.get("response")
        return response is not None and response.varm is not None and view in response.varm

    def _has_required_views(self, views: tuple[str, ...]) -> bool:
        """Check if all required views are available in this dataset."""
        available_mods = set(self._mdata.mod.keys()) - {"response"}
        response = self._mdata.mod.get("response")
        available_varm = set(response.varm.keys()) if response is not None and response.varm is not None else set()
        available_obsm = set(response.obsm.keys()) if response is not None and response.obsm is not None else set()
        available = available_mods | available_varm | available_obsm
        return all(v in available for v in views)

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
