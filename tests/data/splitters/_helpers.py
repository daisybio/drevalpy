"""Shared stubs for the splitter tests.

``MockMuDataset`` is the reference in-memory ``MuDataLike`` stand-in: the
splitters only ever read ``cell_line_ids``, ``drug_ids``, ``response_matrix``
and ``get_tissue``, so a real ``.h5mu`` round-trip buys nothing here.
"""

from __future__ import annotations

import numpy as np

from drevalpy.types import SplitMask


class MockMuDataset:
    """Minimal MuDataLike for testing splitters."""

    def __init__(self, n_cl: int = 10, n_dr: int = 8, density: float = 0.7, n_tissues: int = 3):
        """Build a response matrix with a deterministic pattern of missing values.

        :param n_cl: Number of cell lines (rows).
        :param n_dr: Number of drugs (columns).
        :param density: Fraction of observed entries; the rest become NaN.
        :param n_tissues: Number of distinct tissues cycled over the cell lines.
        """
        rng = np.random.default_rng(42)
        self._response = rng.standard_normal((n_cl, n_dr)).astype(np.float32)
        mask = rng.random((n_cl, n_dr)) > density
        self._response[mask] = np.nan
        self._cl_ids = np.array([f"CL_{i}" for i in range(n_cl)])
        self._dr_ids = np.array([f"DR_{i}" for i in range(n_dr)])
        self._tissues = np.array([f"Tissue_{i % n_tissues}" for i in range(n_cl)])

    @property
    def cell_line_ids(self) -> np.ndarray:
        """Row identifiers of the response matrix."""
        return self._cl_ids

    @property
    def drug_ids(self) -> np.ndarray:
        """Column identifiers of the response matrix."""
        return self._dr_ids

    @property
    def response_matrix(self) -> np.ndarray:
        """Cell-line-by-drug response matrix with NaN for unobserved pairs."""
        return self._response

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        """Return the tissue label for each requested cell line id.

        :param ids: Cell line identifiers to look up.
        :returns: Tissue labels in the order of *ids*.
        """
        idx_map = {name: i for i, name in enumerate(self._cl_ids)}
        indices = [idx_map[str(x)] for x in ids]
        return self._tissues[indices]


def build_mask(shape: tuple[int, int], *positions: tuple[int, int]) -> SplitMask:
    """Build a ``SplitMask`` of *shape* that is True exactly at *positions*.

    :param shape: ``(n_cell_lines, n_drugs)`` shape of the mask.
    :param positions: ``(row, column)`` coordinates to set to True.
    :returns: The assembled ``SplitMask``.
    """
    mask = np.zeros(shape, dtype=bool)
    for row, column in positions:
        mask[row, column] = True
    return SplitMask(mask)
