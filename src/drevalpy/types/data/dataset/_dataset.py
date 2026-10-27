"""MuData-backed dataset class for drevalpy.

``Dataset`` wraps a MuData object and provides typed access to response data,
sample and drug features, metadata, and auxiliary model data backed by an
.h5mu file. Its name and randomization live in ``mdata.uns`` only, so they
survive ``save`` and ``load`` without a second copy.

``mudata`` and ``pandas`` are imported lazily. This module is on the critical
path of ``import drevalpy`` (the registration path reaches it through
:mod:`drevalpy.types`), and ``mudata`` costs ~0.3s because it pulls in
``anndata``, ``dask.array``, ``zarr`` and ``scipy.stats``. Every annotation
referring to them is a string thanks to ``from __future__ import annotations``,
so only the handful of runtime uses need a function-local import. See
``tests/test_import_cost_policy.py``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
from upath import UPath as Path

from drevalpy.types.enums.randomization import randomization_label

from ._dense import to_dense
from ._feature_access import FeatureAccessMixin
from ._io import plain_value, stored_randomization
from ._randomization import RandomizationMixin
from ._sample_column import SampleColumnMixin

if TYPE_CHECKING:
    import mudata as md
    import pandas as pd


def _restrict_replicate_fits(response: md.AnnData) -> None:
    """Keep only the rows of ``response.uns["replicate_fits"]`` for the samples and drugs *response* still has."""
    from drevalpy.data import _replicate_fits

    table = response.uns.get(_replicate_fits.KEY)
    if table is not None:
        response.uns[_replicate_fits.KEY] = _replicate_fits.restrict(table, response.obs_names, response.var_names)


class Dataset(FeatureAccessMixin, RandomizationMixin, SampleColumnMixin):
    """Single entry point for all dataset access in drevalpy.

    Wraps a MuData object containing a "response" modality (sample x drug
    matrix with pEC50 as X) plus any number of sample feature modalities
    (gene_expression, proteomics, etc.). Every other curve metric - ``LN_IC50``,
    ``AUC``, ``IC50``, the goodness-of-fit columns - is a named layer; see
    :func:`drevalpy.curation.curate`.

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

        :param mdata: A MuData object with at least a "response" modality.
        :param name: Human-readable dataset name.
        :param randomization: Optional (mode, view) tuple describing which
            randomization was applied.
        :raises KeyError: If the "response" modality is missing.
        """
        if "response" not in mdata.mod:
            raise KeyError("MuData must contain a 'response' modality.")
        self._mdata = mdata
        mdata.uns["dataset_name"] = name
        self.randomization = randomization

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        """Read a Dataset from an .h5mu file on disk.

        :param path: Path to the .h5mu file.
        :returns: A Dataset wrapping the loaded MuData.
        """
        import mudata as md

        resolved = Path(path)
        with md.set_options(pull_on_update=False):
            mdata = md.read_h5mu(resolved)

        stored_name = mdata.uns.get("dataset_name")
        name = stored_name if isinstance(stored_name, str) else resolved.stem

        if "randomization_provenance" in mdata.uns:
            mdata.uns["randomization_provenance"] = plain_value(mdata.uns["randomization_provenance"])

        return cls(mdata, name=name, randomization=stored_randomization(mdata.uns))

    def save(self, path: str | Path) -> None:
        """Write this Dataset to an .h5mu file.

        :param path: Output file path.
        """
        import mudata as md

        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)

        with md.set_options(pull_on_update=False):
            self._mdata.write(str(resolved))

    @property
    def name(self) -> str:
        """Human-readable dataset name."""
        return str(self._mdata.uns["dataset_name"])

    @property
    def randomization(self) -> tuple[str, str] | None:
        """The ``(mode, view)`` pair of a randomized dataset, or ``None``."""
        return stored_randomization(self._mdata.uns)

    @randomization.setter
    def randomization(self, value: tuple[str, str] | None) -> None:
        if value is None:
            self._mdata.uns.pop("randomization", None)
        else:
            self._mdata.uns["randomization"] = list(value)

    @property
    def mdata(self) -> md.MuData:
        """Return the underlying MuData object."""
        return self._mdata

    # ------------------------------------------------------------------
    # Response access
    # ------------------------------------------------------------------

    @property
    def response(self) -> md.AnnData:
        """Return the response AnnData (samples x drugs)."""
        return self._mdata.mod["response"]

    @property
    def response_matrix(self) -> np.ndarray:
        """pEC50 response matrix (n_samples x n_drugs).

        ``X`` holds pEC50. Use :meth:`get_response_layer` for any other measure,
        for example ``get_response_layer("LN_IC50")``.

        :returns: Dense float32 array of shape (n_samples, n_drugs).
        """
        return np.asarray(to_dense(self.response.X), dtype=np.float32)

    @property
    def sample_ids(self) -> np.ndarray:
        """Sample identifiers (obs_names of the response modality).

        :returns: 1-D string array of sample IDs.
        """
        return np.asarray(self.response.obs_names)

    @property
    def drug_ids(self) -> np.ndarray:
        """Drug identifiers (var_names of the response modality).

        :returns: 1-D string array of PubChem IDs.
        """
        return np.asarray(self.response.var_names)

    def get_response_layer(self, name: str) -> np.ndarray:
        """Retrieve a named response layer (e.g. "AUC").

        :param name: Layer name within the response AnnData.
        :returns: Dense float32 array of shape (n_samples, n_drugs).
        :raises KeyError: If the layer does not exist.
        """
        if name not in self.response.layers:
            raise KeyError(f"Response layer '{name}' not found. Available: {self.response_layer_names()}")
        return np.asarray(to_dense(self.response.layers[name]), dtype=np.float32)

    def response_layer_names(self) -> list[str]:
        """Names of the available response layers.

        :returns: Layer names of the response modality, in insertion order.
        """
        # anndata 0.13 yields a spurious ``None`` from ``layers.keys()``, even for
        # a freshly constructed AnnData with a single layer, so filter to the real
        # names rather than leaking a non-``str`` into a ``list[str]``.
        return [name for name in self.response.layers.keys() if isinstance(name, str)]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def sample_meta(self) -> pd.DataFrame:
        """Global sample metadata (tissue, plus optional cellosaurus_id/cell_line_name for cell-line datasets).

        :returns: DataFrame from mdata.obs, indexed by sample ID.
        """
        return self._mdata.obs

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        """Get tissue labels for the given sample IDs.

        :param ids: 1-D array of sample IDs.
        :returns: 1-D array of tissue labels (NaN for unknown IDs).
        """
        return self.sample_column("tissue", ids)

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------

    def subset_samples(self, ids: np.ndarray) -> Dataset:
        """Return a new Dataset restricted to the given samples.

        Only keeps samples present in the response modality. Other modalities
        are also subset to their intersection with *ids*, and
        ``response.uns["replicate_fits"]`` to the rows of the kept samples.

        :param ids: 1-D array of sample IDs to keep.
        :returns: Independent Dataset containing the selected samples.
        """
        import mudata as md

        ids = np.asarray(ids, dtype=str)
        response_mask = np.isin(self.response.obs_names, ids)
        kept_samples = self.response.obs_names[response_mask]

        new_mods = {
            name: adata[np.isin(adata.obs_names, kept_samples)].copy() for name, adata in self._mdata.mod.items()
        }
        _restrict_replicate_fits(new_mods["response"])

        with md.set_options(pull_on_update=False):
            new_mdata = md.MuData(new_mods)
        new_mdata.obs = self._mdata.obs.loc[self._mdata.obs.index.isin(kept_samples)].copy()
        new_mdata.uns.update(copy.deepcopy(dict(self._mdata.uns)))
        return Dataset(new_mdata, name=self.name, randomization=self.randomization)

    def subset_drugs(self, ids: np.ndarray) -> Dataset:
        """Return a new Dataset restricted to the given drugs.

        Only the response modality has a drug axis; it is subset on var.
        Other modalities (sample features) are kept unchanged.
        ``response.uns["replicate_fits"]`` keeps the rows of the kept drugs.

        :param ids: 1-D array of PubChem drug IDs to keep.
        :returns: Independent Dataset containing the selected drugs.
        """
        import mudata as md

        ids = np.asarray(ids, dtype=str)
        drug_mask = np.isin(self.response.var_names, ids)

        new_mods = {
            name: adata[:, drug_mask].copy() if name == "response" else adata.copy()
            for name, adata in self._mdata.mod.items()
        }
        _restrict_replicate_fits(new_mods["response"])

        with md.set_options(pull_on_update=False):
            new_mdata = md.MuData(new_mods)
        new_mdata.obs = self._mdata.obs.copy()
        new_uns = copy.deepcopy(dict(self._mdata.uns))
        graphs = new_uns.get("drug_graphs")
        if isinstance(graphs, dict):
            retained_drugs = set(map(str, new_mods["response"].var_names))
            new_uns["drug_graphs"] = {drug_id: graph for drug_id, graph in graphs.items() if drug_id in retained_drugs}
        new_mdata.uns.update(new_uns)
        return Dataset(new_mdata, name=self.name, randomization=self.randomization)

    # ------------------------------------------------------------------
    # Auxiliary data
    # ------------------------------------------------------------------

    def get_uns(self, key: str) -> Any:
        """Access arbitrary data from mdata.uns.

        :param key: Key in the uns dict.
        :returns: The stored value.
        :raises KeyError: If the key does not exist.
        """
        if key not in self._mdata.uns:
            raise KeyError(f"uns key '{key}' not found. Available: {list(self._mdata.uns.keys())}")
        return self._mdata.uns[key]

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a formatted summary."""
        n_cl = len(self.sample_ids)
        n_dr = len(self.drug_ids)
        response = self.response_matrix
        n_measured = int(np.sum(~np.isnan(response)))
        mods = [m for m in self._mdata.mod.keys() if m != "response"]

        lines = [
            "Dataset",
            f"    Name: {self.name}",
            f"    Samples: {n_cl}",
            f"    Drugs: {n_dr}",
            f"    Measured pairs: {n_measured}",
            f"    Randomization: {randomization_label(self.randomization) if self.randomization else None}",
            "    Modalities:",
        ]
        for mod in mods:
            shape = self._mdata.mod[mod].X.shape
            lines.append(f"        {mod}: {shape[0]} × {shape[1]}")

        return "\n".join(lines)
