"""Build training and prediction matrices from featurized entity features."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers._matrix import stack_pair_features
from drevalpy.datasets.dataset import DrugResponseDataset


def pair_cell_line_indices(
    cell_line_ids: np.ndarray,
    cell_line_id_to_row: dict[str, int],
) -> np.ndarray:
    return np.array([cell_line_id_to_row[str(cell_id)] for cell_id in cell_line_ids], dtype=np.int64)


def pair_drug_indices(
    drug_ids: np.ndarray,
    drug_id_to_row: dict[str, int],
) -> np.ndarray:
    return np.array([drug_id_to_row[str(drug_id)] for drug_id in drug_ids], dtype=np.int64)


def build_pair_matrix(
    response: DrugResponseDataset,
    cell_line_matrix: np.ndarray,
    drug_matrix: np.ndarray,
    cell_line_entity_ids: np.ndarray,
    drug_entity_ids: np.ndarray,
) -> np.ndarray:
    """Return ``X`` with one row per response pair."""
    cell_line_map = {str(entity_id): row for row, entity_id in enumerate(cell_line_entity_ids)}
    drug_map = {str(entity_id): row for row, entity_id in enumerate(drug_entity_ids)}
    cell_line_idx = pair_cell_line_indices(response.cell_line_ids, cell_line_map)
    if drug_matrix.size == 0:
        return cell_line_matrix[cell_line_idx]
    if cell_line_matrix.size == 0:
        drug_idx = pair_drug_indices(response.drug_ids, drug_map)
        return drug_matrix[drug_idx]
    drug_idx = pair_drug_indices(response.drug_ids, drug_map)
    return stack_pair_features(cell_line_matrix, drug_matrix, cell_line_idx, drug_idx)
