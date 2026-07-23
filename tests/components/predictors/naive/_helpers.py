"""Shared helpers for naive predictor unit tests."""

from __future__ import annotations

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.datasets.dataset import FeatureDataset


def naive_batch(
    *,
    cell_line_ids: np.ndarray,
    drug_ids: np.ndarray,
    response: np.ndarray | None = None,
    cell_line_input: FeatureDataset | None = None,
) -> ModelInputBatch:
    """Build a minimal batch for naive predictor tests."""
    n_pairs = len(cell_line_ids)
    empty = np.array([], dtype=np.float64)
    return ModelInputBatch(
        cell_line_ids=cell_line_ids,
        drug_ids=drug_ids,
        response=response,
        cell_line_entity_ids=cell_line_ids.copy(),
        drug_entity_ids=drug_ids.copy(),
        cell_line_features=empty,
        drug_features=empty,
        cell_line_pair_idx=np.arange(n_pairs, dtype=np.int64),
        drug_pair_idx=np.arange(n_pairs, dtype=np.int64),
        cell_line_input=cell_line_input,
    )
