"""Core data structure types."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class EntityScope:
    """Index scope for model train/predict operations.

    Unlike SplitMasks (which represents a CV fold partition), EntityScope simply
    says "operate on these entities." Used by _ComponentStack.train() and predict().
    """

    cell_lines: np.ndarray
    drugs: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class SplitMasks:
    """Index arrays for a single cross-validation fold.

    For LCO/LTO the drug indices are *None* (all drugs used for all splits).
    For LDO the cell line indices cover all cell lines and drug indices differ.
    For LPO both cell_line and drug indices are populated (paired).
    """

    train_cell_lines: np.ndarray
    test_cell_lines: np.ndarray
    val_cell_lines: np.ndarray

    train_drugs: np.ndarray | None = None
    test_drugs: np.ndarray | None = None
    val_drugs: np.ndarray | None = None
