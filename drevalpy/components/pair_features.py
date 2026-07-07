"""Index helpers for mapping response pairs to featurizer entity rows."""

from __future__ import annotations

import numpy as np


def pair_cell_line_indices(
    cell_line_ids: np.ndarray,
    cell_line_id_to_row: dict[str, int],
) -> np.ndarray:
    """Map pair cell-line identifiers to featurizer row indices."""
    return np.array([cell_line_id_to_row[str(cell_id)] for cell_id in cell_line_ids], dtype=np.int64)


def pair_drug_indices(
    drug_ids: np.ndarray,
    drug_id_to_row: dict[str, int],
) -> np.ndarray:
    """Map pair drug identifiers to featurizer row indices."""
    return np.array([drug_id_to_row[str(drug_id)] for drug_id in drug_ids], dtype=np.int64)
