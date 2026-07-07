"""Helpers for reading identity metadata from structured pair batches."""

from __future__ import annotations

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.utils import TISSUE_IDENTIFIER


def _labels_from_one_hot(
    one_hot: np.ndarray,
    categories: np.ndarray,
    pair_idx: np.ndarray,
) -> np.ndarray:
    pair_rows = np.asarray(one_hot)[pair_idx]
    if pair_rows.ndim == 1:
        pair_rows = pair_rows.reshape(-1, 1)
    if pair_rows.size == 0:
        return np.array([], dtype=str)
    indices = np.argmax(pair_rows, axis=1)
    category_list = [str(category) for category in np.asarray(categories).reshape(-1)]
    return np.array(
        [category_list[index] if pair_rows[row, index] > 0 else "" for row, index in enumerate(indices)],
        dtype=str,
    )


def _tissue_labels_from_input(
    batch: ModelInputBatch,
    cell_line_input: FeatureDataset,
) -> np.ndarray | None:
    if not any(TISSUE_IDENTIFIER in views for views in cell_line_input.features.values()):
        return None
    tissues = cell_line_input.get_feature_matrix(view=TISSUE_IDENTIFIER, identifiers=batch.cell_line_ids)
    return np.asarray(tissues).reshape(-1).astype(str)


def pair_tissue_ids(
    batch: ModelInputBatch,
    *,
    cell_line_input: FeatureDataset | None = None,
) -> np.ndarray | None:
    """Return tissue labels aligned with each pair in *batch*, when available."""
    tissue_block = batch.cell_line_blocks.get("tissue")
    categories = batch.cell_line_blocks.get("tissue_categories")
    if tissue_block is not None and categories is not None and np.asarray(tissue_block).ndim == 2:
        labels = _labels_from_one_hot(tissue_block, categories, batch.cell_line_pair_idx)
        if labels.size and np.any(labels != ""):
            return labels
    if tissue_block is not None and np.asarray(tissue_block).dtype == object:
        values = np.asarray(tissue_block[batch.cell_line_pair_idx]).reshape(-1)
        return np.array(
            [str(value.item() if isinstance(value, np.ndarray) else value) for value in values],
            dtype=str,
        )
    if cell_line_input is not None:
        return _tissue_labels_from_input(batch, cell_line_input)
    return None
