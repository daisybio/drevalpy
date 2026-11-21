"""In-memory ``Dataset`` stand-in for tests of the quality rule and the splitters.

``drevalpy.synthetic()`` costs about 0.3s even at 10x8, too much for fast-tier tests; this fake serves the
same surface (ids, response matrix, sample columns, response layers) from arrays built in microseconds. It
lives at the root of ``tests/`` because its users span ``tests/data/`` and ``tests/registry/splitter/``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


class FakeResponseDataset:
    """A ``Dataset`` whose response matrix, sample columns and response layers are set by the test.

    Sample ids are ``CL_{i}`` and drug ids ``D_{i}``. Unless a test says otherwise, the response matrix is
    random with ``1 - density`` of its entries missing, the tissues cycle over three labels, and the
    ``relevance_score`` and ``fold_change`` layers pass the default quality rule.
    """

    #: ``CurveQuality`` reads the primary measure through ``dataset.x_measure``.
    x_measure = "pEC50"
    measure = "pEC50"

    def __init__(
        self,
        response: np.ndarray | None = None,
        *,
        shape: tuple[int, int] = (10, 8),
        density: float = 0.7,
        tissues: Sequence[str] | None = None,
        columns: Mapping[str, Sequence[str]] | None = None,
        layers: Mapping[str, np.ndarray] | None = None,
        failing_pairs: Iterable[tuple[int, int]] = (),
    ) -> None:
        """Build the matrix, the sample columns and the layers.

        :param response: Explicit response matrix; when ``None``, a seeded random matrix of *shape* in which
            each entry is observed with probability *density*.
        :param shape: Number of samples and drugs of the random matrix; ignored when *response* is given.
        :param density: Fraction of the random matrix that is observed.
        :param tissues: The ``tissue`` column, one label per sample; defaults to three labels cycling over
            the samples.
        :param columns: Further per-sample columns, such as ``study`` or ``source_sample_id``.
        :param layers: The response layers; ``{}`` gives a dataset without quality layers. Defaults to
            ``relevance_score`` and ``fold_change`` layers that pass the default rule.
        :param failing_pairs: ``(sample row, drug column)`` pairs that fail the default layers; only valid
            together with the default *layers*.
        :raises ValueError: If *failing_pairs* is given together with explicit *layers*.
        """
        if response is None:
            rng = np.random.default_rng(42)
            response = rng.standard_normal(shape).astype(np.float32)
            response[rng.random(shape) > density] = np.nan
        self._response = response
        n_samples, n_drugs = response.shape
        self._sample_ids = np.array([f"CL_{i}" for i in range(n_samples)])
        self._drug_ids = np.array([f"D_{i}" for i in range(n_drugs)])
        if tissues is None:
            tissues = [f"Tissue_{i % 3}" for i in range(n_samples)]
        self._columns = {
            "tissue": np.array(tissues),
            **{name: np.array(values) for name, values in (columns or {}).items()},
        }

        failing_pairs = list(failing_pairs)
        if layers is not None and failing_pairs:
            raise ValueError("failing_pairs only applies to the default layers; build the failing layers explicitly.")
        self._layers = dict(layers) if layers is not None else _passing_layers(response.shape, failing_pairs)

    @property
    def sample_ids(self) -> np.ndarray:
        """Row identifiers of the response matrix."""
        return self._sample_ids

    @property
    def drug_ids(self) -> np.ndarray:
        """Column identifiers of the response matrix."""
        return self._drug_ids

    @property
    def response_matrix(self) -> np.ndarray:
        """Sample-by-drug response matrix with NaN for unobserved pairs."""
        return self._response

    def sample_column(self, column: str, ids: np.ndarray) -> np.ndarray:
        """Return *column* for each requested sample id."""
        if column not in self._columns:
            raise KeyError(f"Sample column '{column}' not found. Available: {list(self._columns)}")
        rows = {sample_id: row for row, sample_id in enumerate(self._sample_ids)}
        return self._columns[column][[rows[str(sample_id)] for sample_id in ids]]

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        """Return the tissue label for each requested sample id."""
        return self.sample_column("tissue", ids)

    def response_layer_names(self) -> list[str]:
        """Names of the available response layers."""
        return list(self._layers)

    def get_response_layer(self, name: str) -> np.ndarray:
        """Return a named response layer; the primary measure is the response matrix itself."""
        if name == self.x_measure:
            return self._response
        if name not in self._layers:
            raise KeyError(f"Response layer '{name}' not found. Available: {list(self._layers)}")
        return self._layers[name]


def _passing_layers(shape: tuple[int, int], failing_pairs: Iterable[tuple[int, int]]) -> dict[str, np.ndarray]:
    """The ``relevance_score`` and ``fold_change`` layers of the default rule, zeroed at *failing_pairs*."""
    relevance = np.full(shape, 9.0, dtype=np.float32)
    fold_change = np.full(shape, -2.0, dtype=np.float32)
    for row, column in failing_pairs:
        relevance[row, column] = 0.0
        fold_change[row, column] = 0.0
    return {"relevance_score": relevance, "fold_change": fold_change}
