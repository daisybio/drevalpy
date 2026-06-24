"""Rebuild :class:`FeatureDataset` objects from structured featurizer blocks."""

from __future__ import annotations

import numpy as np

from drevalpy.datasets.dataset import FeatureDataset


def feature_dataset_from_blocks(
    entity_ids: np.ndarray,
    blocks: dict[str, np.ndarray],
    *,
    fallback: FeatureDataset | None = None,
) -> FeatureDataset:
    """Materialize per-entity view dicts from named featurizer blocks."""
    if not blocks:
        if fallback is None:
            msg = "structured batch has no cell-line blocks and no fallback dataset"
            raise ValueError(msg)
        return fallback
    features: dict[str, dict[str, np.ndarray]] = {}
    for row, entity_id in enumerate(entity_ids):
        entity_key = str(entity_id)
        features[entity_key] = {view: np.asarray(matrix[row], dtype=np.float32) for view, matrix in blocks.items()}
    return FeatureDataset(features)


def merge_feature_dataset(
    primary: FeatureDataset,
    blocks: dict[str, np.ndarray],
    entity_ids: np.ndarray,
) -> FeatureDataset:
    """Overlay structured block views onto an existing feature dataset."""
    if not blocks:
        return primary
    merged = {entity: dict(views) for entity, views in primary.features.items()}
    for row, entity_id in enumerate(entity_ids):
        entity_key = str(entity_id)
        if entity_key not in merged:
            merged[entity_key] = {}
        for view, matrix in blocks.items():
            merged[entity_key][view] = np.asarray(matrix[row], dtype=np.float32)
    return FeatureDataset(merged)
