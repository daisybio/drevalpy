"""Rebuild `FeatureDataset` objects from structured featurizer blocks."""

from __future__ import annotations

import numpy as np

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.feature_block import FeatureBlock
from drevalpy.datasets.dataset import FeatureDataset


def _materialize_row(block: FeatureBlock, row: int) -> np.ndarray:
    value = block.values[row]
    if block.format == FeatureFormat.NUMERIC_MATRIX:
        return np.asarray(value, dtype=np.float32)
    return value


def feature_dataset_from_blocks(
    entity_ids: np.ndarray,
    blocks: dict[str, FeatureBlock],
    *,
    fallback: FeatureDataset | None = None,
) -> FeatureDataset:
    """Materialize per-entity view dicts from named featurizer blocks."""
    entity_blocks = {name: block for name, block in blocks.items() if block.entity_aligned}
    if not entity_blocks:
        if fallback is None:
            msg = "structured batch has no cell-line blocks and no fallback dataset"
            raise ValueError(msg)
        return fallback

    meta_info: dict[str, list[str]] = {}
    for view, block in entity_blocks.items():
        if block.feature_names is not None:
            meta_info[view] = list(block.feature_names)

    features: dict[str, dict[str, np.ndarray]] = {}
    for row, entity_id in enumerate(entity_ids):
        entity_key = str(entity_id)
        features[entity_key] = {view: _materialize_row(block, row) for view, block in entity_blocks.items()}
    return FeatureDataset(features, meta_info=meta_info)


def merge_feature_dataset(
    primary: FeatureDataset,
    blocks: dict[str, FeatureBlock],
    entity_ids: np.ndarray,
) -> FeatureDataset:
    """Overlay structured block views onto an existing feature dataset."""
    entity_blocks = {name: block for name, block in blocks.items() if block.entity_aligned}
    if not entity_blocks:
        return primary
    merged = {entity: dict(views) for entity, views in primary.features.items()}
    meta_info = dict(primary.meta_info)
    for view, block in entity_blocks.items():
        if block.feature_names is not None:
            meta_info[view] = list(block.feature_names)
    for row, entity_id in enumerate(entity_ids):
        entity_key = str(entity_id)
        if entity_key not in merged:
            merged[entity_key] = {}
        for view, block in entity_blocks.items():
            merged[entity_key][view] = _materialize_row(block, row)
    return FeatureDataset(merged, meta_info=meta_info)
