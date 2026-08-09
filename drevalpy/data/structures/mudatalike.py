"""Protocol for MuDataset-compatible objects."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class MuDataLike(Protocol):
    """Minimal interface expected from a MuDataset-compatible object.

    This allows splitters and other components to work with the real MuDataset
    or with any object satisfying the protocol for testing.
    """

    @property
    def cell_line_ids(self) -> np.ndarray:
        """1-D array of cell line identifiers (obs_names of the response modality)."""
        ...

    @property
    def drug_ids(self) -> np.ndarray:
        """1-D array of drug identifiers (var_names of the response modality)."""
        ...

    @property
    def response_matrix(self) -> np.ndarray:
        """2-D float array (n_cell_lines x n_drugs). NaN where no measurement."""
        ...

    def get_tissue(self, ids: np.ndarray) -> np.ndarray:
        """Return tissue labels for the given cell line IDs."""
        ...
