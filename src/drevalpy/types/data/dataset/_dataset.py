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

from typing import TYPE_CHECKING, Any

import numpy as np
from upath import UPath as Path

from drevalpy.types.data.views import TISSUE_IDENTIFIER
from drevalpy.types.enums.randomization import RandomizationType, randomization_label

from ._anndata import layer_names, to_dense
from ._io import plain_value, stored_randomization
from ._randomization import randomized_mdata

if TYPE_CHECKING:
    import mudata as md
    import pandas as pd


class Dataset:
    """Single entry point for all dataset access in drevalpy.

    Wraps a MuData object containing a "response" modality (sample x drug
    matrix with pEC50 as X) plus any number of sample feature modalities
    (gene_expression, proteomics, etc.). Every other curve metric - ``LN_IC50``,
    ``AUC``, ``IC50``, the goodness-of-fit columns - is a named layer; see
    :func:`drevalpy.curation.curate`.

    Drug features are stored as ``response.varm`` entries, drug graphs in
    ``mdata.uns["drug_graphs"]``, and model-specific auxiliary data in other
    ``mdata.uns`` keys. Featurizers read the views through a
    :class:`~drevalpy.types.data.feature_source.FeatureSource`.
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

        dataset = cls(mdata, name=name, randomization=stored_randomization(mdata.uns))
        if dataset.randomization_provenance is not None:
            dataset.randomization_provenance = plain_value(dataset.randomization_provenance)
        return dataset

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
    def randomization_provenance(self) -> dict[str, Any] | None:
        """Seeds, type and views of the randomization test that built this dataset, or ``None``."""
        return self._mdata.uns.get("randomization_provenance")

    @randomization_provenance.setter
    def randomization_provenance(self, value: dict[str, Any]) -> None:
        self._mdata.uns["randomization_provenance"] = value

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
        return layer_names(self.response)

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
        return self.sample_column(TISSUE_IDENTIFIER, ids)

    def sample_column(self, column: str, ids: np.ndarray) -> np.ndarray:
        """Return one ``obs`` column for the given sample IDs.

        :param column: Name of a column of the sample metadata (``obs``), such as
            ``"tissue"`` or ``"study"``.
        :param ids: 1-D array of sample IDs.
        :returns: 1-D array with one value per requested ID, in the order of *ids*.
            IDs that are not samples of this dataset give NaN.
        :raises KeyError: If *column* is not in ``obs``; the message lists the available columns.
        """
        obs = self._mdata.obs
        if column not in obs.columns:
            raise KeyError(f"Sample column '{column}' not found. Available: {list(obs.columns)}")
        return obs[column].reindex(np.asarray(ids, dtype=str)).to_numpy()

    # ------------------------------------------------------------------
    # Randomization
    # ------------------------------------------------------------------

    def with_randomized_views(
        self,
        views: list[str],
        randomization_type: RandomizationType = RandomizationType.PERMUTATION,
        random_state: int | None = None,
        *,
        randomization: tuple[str, str] | None = None,
    ) -> Dataset:
        """Return a copy of this Dataset with specified views randomized.

        For sample views (modalities or obsm keys), rows are permuted across
        samples. For drug views (varm keys), rows are permuted across drugs.
        For uns dict keys, values are reassigned to shuffled keys.

        Raises ``KeyError`` if a view is not found in any storage location.

        :param views: View names to randomize.
        :param randomization_type: ``permutation`` shuffles rows; ``invariant`` replaces
            each row with a random draw matching its mean and std. A value that is
            not a :class:`RandomizationType` is rejected.
        :param random_state: Seed for reproducibility.
        :param randomization: Optional (mode, view) tuple to attach to the new dataset.
        :returns: A new Dataset with the specified views randomized.
        """
        new_mdata = randomized_mdata(self._mdata, views, RandomizationType(randomization_type), random_state)
        return Dataset(new_mdata, name=self.name, randomization=randomization)

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
