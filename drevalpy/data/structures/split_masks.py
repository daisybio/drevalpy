"""Index arrays for a single cross-validation fold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
