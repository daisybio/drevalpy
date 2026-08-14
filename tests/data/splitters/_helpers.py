"""Shared stubs for the splitter tests.

``MockMuDataset`` is the reference in-memory ``MuDataLike`` stand-in: the
splitters read ``cell_line_ids``, ``drug_ids``, ``response_matrix``,
``get_tissue`` and the response layers behind
:func:`drevalpy.data.quality.curve_quality_mask`, so a real ``.h5mu``
round-trip buys nothing here.

By default every curve passes the quality filter, which keeps the folds a
splitter produces determined purely by ``response_matrix``. Pass
*failing_pairs* to mark individual pairs as low quality.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from drevalpy.types import SplitMask


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
        """Build a response matrix with a deterministic pattern of missing values.

        :param n_cl: Number of cell lines (rows).
        :param n_dr: Number of drugs (columns).
        :param density: Fraction of observed entries; the rest become NaN.
        :param n_tissues: Number of distinct tissues cycled over the cell lines.
        :param failing_pairs: ``(row, column)`` pairs to mark as failing the
            default curve-quality thresholds.
        """
        rng = np.random.default_rng(42)
        self._response = rng.standard_normal((n_cl, n_dr)).astype(np.float32)
        mask = rng.random((n_cl, n_dr)) > density
        self._response[mask] = np.nan
        self._cl_ids = np.array([f"CL_{i}" for i in range(n_cl)])
        self._dr_ids = np.array([f"DR_{i}" for i in range(n_dr)])
        self._tissues = np.array([f"Tissue_{i % n_tissues}" for i in range(n_cl)])

        # Comfortably above/below the default thresholds, so a boundary change
        # in the filter cannot silently flip these fixtures.
        relevance = np.full((n_cl, n_dr), 9.0, dtype=np.float32)
        fold_change = np.full((n_cl, n_dr), -2.0, dtype=np.float32)
        for row, column in failing_pairs:
            relevance[row, column] = 0.0
            fold_change[row, column] = 0.0
        self._layers = {"relevance_score": relevance, "fold_change": fold_change}

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

    def response_layer_names(self) -> list[str]:
        """Names of the available response layers."""
        return list(self._layers)

    def get_response_layer(self, name: str) -> np.ndarray:
        """Return a named response layer.

        :param name: Layer name.
        :returns: Cell-line-by-drug matrix for that layer.
        :raises KeyError: If the layer was not built.
        """
        if name not in self._layers:
            raise KeyError(f"Response layer '{name}' not found. Available: {list(self._layers)}")
        return self._layers[name]


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


def first_measured_pairs(dataset: MockMuDataset, count: int) -> list[tuple[int, int]]:
    """Return the first *count* measured ``(row, column)`` pairs of *dataset*.

    Used to pick pairs that the quality filter can visibly remove: blanking an
    already-unmeasured pair would prove nothing.
    """
    measured = np.argwhere(~np.isnan(dataset.response_matrix))
    return [(int(row), int(column)) for row, column in measured[:count]]


def covered_pairs(folds: list) -> np.ndarray:
    """Union every train, test and validation mask across *folds*.

    A pair the splitter considers usable lands in at least one of them, so this
    is what a quality-filtered pair must be absent from.
    """
    covered = np.zeros(folds[0].train.shape, dtype=bool)
    for fold in folds:
        covered |= fold.train.mask | fold.test.mask | fold.val.mask
    return covered
