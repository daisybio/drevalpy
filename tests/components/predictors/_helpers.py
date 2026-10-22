"""Shared batch builders for predictor unit tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from drevalpy.components.contracts.training_context import TrainingContext
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from tests.models.synthetic_fixtures import (
    drug_fingerprints,
    multi_drug_response,
    sample_gene_expression,
)


def neural_batch(*, with_early_stopping: bool = False) -> ModelInputBatch:
    """Build a dense matrix batch for predictors that train on flattened features.

    :param with_early_stopping: Attach a validation :class:`ResponseBatch` when true.
    :returns: Featurized ``ModelInputBatch`` with a temporary checkpoint directory.
    """
    response = multi_drug_response()
    sample_features = np.vstack(
        [
            sample_gene_expression().features["cl1"]["gene_expression"],
            sample_gene_expression().features["cl2"]["gene_expression"],
        ]
    )
    drug_features = np.vstack(
        [
            drug_fingerprints().features["d1"]["fingerprints"],
            drug_fingerprints().features["d2"]["fingerprints"],
        ]
    )
    early_stopping = None
    if with_early_stopping:
        early_stopping = ResponseBatch(
            response=np.array([1.5, 2.5]),
            sample_ids=np.array(["cl1", "cl2"]),
            drug_ids=np.array(["d1", "d2"]),
        )
    return ModelInputBatch.from_response(
        response,
        sample=FeaturizerResult.from_blocks(
            {"gene_expression": numeric_feature_block(sample_features)}, np.array(["cl1", "cl2"])
        ),
        drug=FeaturizerResult.from_blocks(
            {"fingerprints": numeric_feature_block(drug_features)}, np.array(["d1", "d2"])
        ),
        sample_pair_idx=np.array([0, 0, 1, 1]),
        drug_pair_idx=np.array([0, 1, 0, 1]),
        early_stopping_response=early_stopping,
        training_context=TrainingContext(checkpoint_dir=Path(tempfile.mkdtemp())),
    )
