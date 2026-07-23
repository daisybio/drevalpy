"""Build `ModelInputBatch` from featurizer outputs."""

from __future__ import annotations

import numpy as np

from drevalpy.components.model_input_batch import ModelInputBatch
from drevalpy.components.pair_features import pair_cell_line_indices, pair_drug_indices
from drevalpy.components.training_context import TrainingContext
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


def _validate_entity_feature_alignment(
    entity_ids: np.ndarray,
    features: np.ndarray | None,
    *,
    side: str,
) -> None:
    if features is None or features.size == 0:
        return
    if entity_ids.size == 0:
        msg = f"{side}_entity_ids must be non-empty when {side}_features are present"
        raise ValueError(msg)
    if features.ndim == 0 or features.shape[0] != entity_ids.size:
        msg = (
            f"{side}_entity_ids length ({entity_ids.size}) must match "
            f"{side}_features rows ({0 if features.ndim == 0 else features.shape[0]})"
        )
        raise ValueError(msg)


def build_model_input_batch(
    response: DrugResponseDataset,
    *,
    cell_line_entity_ids: np.ndarray,
    drug_entity_ids: np.ndarray | None,
    cell_line_features: np.ndarray,
    drug_features: np.ndarray | None,
    cell_line_blocks: dict[str, np.ndarray] | None = None,
    drug_blocks: dict[str, np.ndarray] | None = None,
    cell_line_input: FeatureDataset | None = None,
    drug_input: FeatureDataset | None = None,
    early_stopping_response: DrugResponseDataset | None = None,
    training_context: TrainingContext | None = None,
) -> ModelInputBatch:
    """Index entity-level featurizer outputs for each response pair."""
    n_pairs = len(response)
    _validate_entity_feature_alignment(cell_line_entity_ids, cell_line_features, side="cell_line")
    if drug_entity_ids is not None:
        _validate_entity_feature_alignment(drug_entity_ids, drug_features, side="drug")

    if cell_line_entity_ids.size == 0:
        if n_pairs > 0 and cell_line_features.size != 0:
            msg = "cell_line_entity_ids must be non-empty when cell_line_features are present"
            raise ValueError(msg)
        cell_line_pair_idx = np.zeros(n_pairs, dtype=np.int64)
    else:
        cell_line_map = {str(entity_id): row for row, entity_id in enumerate(cell_line_entity_ids)}
        cell_line_pair_idx = pair_cell_line_indices(response.cell_line_ids, cell_line_map)

    drug_pair_idx = None
    if drug_entity_ids is not None and drug_features is not None:
        if drug_entity_ids.size == 0:
            if n_pairs > 0 and drug_features.size != 0:
                msg = "drug_entity_ids must be non-empty when drug_features are present"
                raise ValueError(msg)
            drug_pair_idx = np.zeros(n_pairs, dtype=np.int64)
        else:
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
        cell_line_input=cell_line_input,
        drug_input=drug_input,
        early_stopping_response=early_stopping_response,
        training_context=training_context,
    )
