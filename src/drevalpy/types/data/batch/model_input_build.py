"""Map response positions to already validated canonical entity feature results."""

from __future__ import annotations

import numpy as np

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.types.data.batch.feature_block import FeaturizerResult
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch, _map_pair_indices
from drevalpy.types.data.batch.response_batch import ResponseBatch


def _indices(table: FeaturizerResult, ids: np.ndarray, side: str) -> np.ndarray:
    mapping = {str(entity): row for row, entity in enumerate(table.entity_ids)}
    if len(mapping) != len(table.entity_ids):
        raise ValueError(f"{side} entity_ids must be unique in a predictor entity table")
    return _map_pair_indices(ids, mapping, side=side)


def build_model_input_batch(
    response: ResponseBatch,
    *,
    sample: FeaturizerResult,
    drug: FeaturizerResult | None = None,
    early_stopping_response: ResponseBatch | None = None,
    training_context: TrainingContext | None = None,
) -> ModelInputBatch:
    """Map pair identifiers once, retaining each canonical feature result.

    :param response: Requested pair IDs and targets.
    :param sample: Sample entity results.
    :param drug: Optional drug entity results.
    :param early_stopping_response: Optional disjoint stopping pairs.
    :param training_context: Runtime metadata.
    :returns: Predictor batch in response order.
    """
    return ModelInputBatch.from_response(
        response,
        sample=sample,
        drug=drug,
        sample_pair_idx=_indices(sample, response.sample_ids, "sample"),
        drug_pair_idx=None if drug is None else _indices(drug, response.drug_ids, "drug"),
        early_stopping_response=early_stopping_response,
        training_context=training_context,
    )
