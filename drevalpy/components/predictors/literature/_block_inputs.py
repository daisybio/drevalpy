"""Materialize featurizer blocks into FeatureDataset views."""

from __future__ import annotations

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.predictors.literature._feature_dataset_from_batch import feature_dataset_from_blocks
from drevalpy.datasets.dataset import FeatureDataset


def materialize_block_inputs(
    predictor: object,
    batch: ModelInputBatch,
    *,
    required_cell_line_blocks: tuple[str, ...],
    required_drug_blocks: tuple[str, ...],
    requires_drug_featurizer: bool,
) -> tuple[FeatureDataset, FeatureDataset | None]:
    """Build cell-line and optional drug FeatureDatasets from batch blocks."""
    cell_lines = _dataset_from_blocks(
        predictor,
        batch.cell_line_entity_ids,
        batch.cell_line_blocks,
        required=required_cell_line_blocks,
        side="cell_line",
    )
    if not requires_drug_featurizer:
        return cell_lines, None
    drugs = _dataset_from_blocks(
        predictor,
        batch.drug_entity_ids,
        batch.drug_blocks,
        required=required_drug_blocks,
        side="drug",
    )
    return cell_lines, drugs


def _dataset_from_blocks(
    predictor: object,
    entity_ids: np.ndarray | None,
    blocks: dict[str, np.ndarray],
    *,
    required: tuple[str, ...],
    side: str,
) -> FeatureDataset:
    if entity_ids is None:
        msg = f"{predictor.__class__.__name__} requires {side} entity ids"
        raise RuntimeError(msg)
    missing = [name for name in required if name not in blocks]
    if missing:
        msg = f"{predictor.__class__.__name__} missing {side} block(s) {missing}; required={list(required)}"
        raise ValueError(msg)
    selected = {name: blocks[name] for name in required}
    return feature_dataset_from_blocks(entity_ids, selected)
