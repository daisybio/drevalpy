"""Index helpers for mapping response pairs to featurizer entity rows."""

from __future__ import annotations

import numpy as np


def _map_pair_indices(
    entity_ids: np.ndarray,
    id_to_row: dict[str, int],
    *,
    side: str,
) -> np.ndarray:
    """Map pair identifiers to featurizer row indices with contextual errors."""
    missing: list[str] = []
    rows: list[int] = []
    for entity_id in entity_ids:
        key = str(entity_id)
        row = id_to_row.get(key)
        if row is None:
            missing.append(key)
        else:
            rows.append(row)
    if missing:
        preview = ", ".join(repr(item) for item in missing[:5])
        suffix = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        msg = f"Missing {side} identifiers in featurizer rows: {preview}{suffix}"
        raise ValueError(msg)
    return np.asarray(rows, dtype=np.int64)


def pair_cell_line_indices(
    cell_line_ids: np.ndarray,
    cell_line_id_to_row: dict[str, int],
) -> np.ndarray:
    """Map pair cell-line identifiers to featurizer row indices.

    Args:
        cell_line_ids: Cell-line id per response pair.
        cell_line_id_to_row: Mapping from entity id to featurizer row index.

    Returns:
        Integer array of row indices aligned with *cell_line_ids*.

    Raises:
        ValueError: If any pair id is missing from *cell_line_id_to_row*.
    """
    return _map_pair_indices(cell_line_ids, cell_line_id_to_row, side="cell-line")


def pair_drug_indices(
    drug_ids: np.ndarray,
    drug_id_to_row: dict[str, int],
) -> np.ndarray:
    """Map pair drug identifiers to featurizer row indices.

    Args:
        drug_ids: Drug id per response pair.
        drug_id_to_row: Mapping from entity id to featurizer row index.

    Returns:
        Integer array of row indices aligned with *drug_ids*.

    Raises:
        ValueError: If any pair id is missing from *drug_id_to_row*.
    """
    return _map_pair_indices(drug_ids, drug_id_to_row, side="drug")
