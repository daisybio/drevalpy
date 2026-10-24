"""In-memory ``MuDataLike`` stand-in for splitter tests; all curves pass quality unless *failing_pairs* says not."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class MockMuDataset:
    """Minimal MuDataLike for testing splitters."""

    def __init__(
        self,
        n_cl: int = 10,
        n_dr: int = 8,
        density: float = 0.7,
        n_tissues: int = 3,
        failing_pairs: Iterable[tuple[int, int]] = (),
    ):
        """Build a response matrix with a deterministic pattern of missing values."""
        rng = np.random.default_rng(42)
        self._response = rng.standard_normal((n_cl, n_dr)).astype(np.float32)
        mask = rng.random((n_cl, n_dr)) > density
        self._response[mask] = np.nan
        self._cl_ids = np.array([f"CL_{i}" for i in range(n_cl)])
        self._dr_ids = np.array([f"DR_{i}" for i in range(n_dr)])
        self._tissues = np.array([f"Tissue_{i % n_tissues}" for i in range(n_cl)])

        relevance = np.full((n_cl, n_dr), 9.0, dtype=np.float32)
        fold_change = np.full((n_cl, n_dr), -2.0, dtype=np.float32)
        for row, column in failing_pairs:
            relevance[row, column] = 0.0
            fold_change[row, column] = 0.0
        self._layers = {"relevance_score": relevance, "fold_change": fold_change}

    @property
    def sample_ids(self) -> np.ndarray:
        """Row identifiers of the response matrix."""
        return self._cl_ids

    @property
    def drug_ids(self) -> np.ndarray:
        """Column identifiers of the response matrix."""
        return self._dr_ids

    @property
    def response_matrix(self) -> np.ndarray:
        """Sample-by-drug response matrix with NaN for unobserved pairs."""
        return self._response

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        """Return the tissue label for each requested sample id."""
        idx_map = {name: i for i, name in enumerate(self._cl_ids)}
        indices = [idx_map[str(x)] for x in ids]
        return self._tissues[indices]

    def sample_column(self, column: str, ids: np.ndarray) -> np.ndarray:
        """Serve the tissue column, the only per-sample metadata this fake has."""
        if column != "tissue":
            raise KeyError(f"Sample column '{column}' not found. Available: ['tissue']")
        return self.get_tissue(ids)

    def response_layer_names(self) -> list[str]:
        """Names of the available response layers."""
        return list(self._layers)

    def get_response_layer(self, name: str) -> np.ndarray:
        """Return a named response layer."""
        if name not in self._layers:
            raise KeyError(f"Response layer '{name}' not found. Available: {list(self._layers)}")
        return self._layers[name]


def first_measured_pairs(dataset: MockMuDataset, count: int) -> list[tuple[int, int]]:
    """Return the first *count* measured ``(row, column)`` pairs of *dataset*."""
    measured = np.argwhere(~np.isnan(dataset.response_matrix))
    return [(int(row), int(column)) for row, column in measured[:count]]


def covered_pairs(folds: list) -> np.ndarray:
    """Union every train, test and validation mask across *folds*."""
    covered = np.zeros(folds[0].train.shape, dtype=bool)
    for fold in folds:
        covered |= fold.train.mask | fold.test.mask | fold.val.mask
    return covered
