"""Build `ModelInputBatch` from featurizer outputs."""

from __future__ import annotations

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.pair_features import pair_cell_line_indices, pair_drug_indices
from drevalpy.datasets.dataset import DrugResponseDataset


def build_model_input_batch(
    response: DrugResponseDataset,
    *,
    cell_line_entity_ids: np.ndarray,
    drug_entity_ids: np.ndarray | None,
    cell_line_features: np.ndarray,
    drug_features: np.ndarray | None,
    cell_line_blocks: dict[str, np.ndarray] | None = None,
    drug_blocks: dict[str, np.ndarray] | None = None,
) -> ModelInputBatch:
    """Index entity-level featurizer outputs for each response pair."""
    cell_line_map = {str(entity_id): row for row, entity_id in enumerate(cell_line_entity_ids)}
    cell_line_pair_idx = pair_cell_line_indices(response.cell_line_ids, cell_line_map)
    drug_pair_idx = None
    if drug_entity_ids is not None and drug_features is not None:
        drug_map = {str(entity_id): row for row, entity_id in enumerate(drug_entity_ids)}
        drug_pair_idx = pair_drug_indices(response.drug_ids, drug_map)
    return ModelInputBatch.from_response(
        response,
        cell_line_entity_ids=cell_line_entity_ids,
        drug_entity_ids=drug_entity_ids,
        cell_line_features=cell_line_features,
        drug_features=drug_features,
        cell_line_pair_idx=cell_line_pair_idx,
        drug_pair_idx=drug_pair_idx,
        cell_line_blocks=cell_line_blocks,
        drug_blocks=drug_blocks,
    )
