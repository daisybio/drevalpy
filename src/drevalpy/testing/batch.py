"""Synthetic :class:`ModelInputBatch` construction for predictor tests.

Predictors consume a fully featurized batch, so exercising one normally means
composing a whole model. This module short-circuits that: it draws dense feature
matrices directly and pairs them with the measured entries of a dataset's
response matrix, so a predictor can be trained without any featurizer at all.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.dataset import Dataset

N_SAMPLE_FEATURES = 8
N_DRUG_FEATURES = 4
SEED = 20260813


def observed_pairs(dataset: Dataset) -> ResponseBatch:
    """Return every measured sample/drug pair in *dataset*.

    Args:
        dataset: Dataset whose response matrix is read.

    Returns:
        Response triples for the non-NaN entries, in row-major order.
    """
    matrix = dataset.response_matrix
    rows, columns = np.nonzero(~np.isnan(matrix))
    return ResponseBatch(
        response=matrix[rows, columns].astype(np.float64),
        sample_ids=np.asarray(dataset.sample_ids)[rows],
        drug_ids=np.asarray(dataset.drug_ids)[columns],
    )


def build_synthetic_batch(
    dataset: Dataset,
    *,
    sample_block_names: Sequence[str] = (),
    drug_block_names: Sequence[str] = (),
    n_sample_features: int = N_SAMPLE_FEATURES,
    n_drug_features: int | None = N_DRUG_FEATURES,
    seed: int = SEED,
) -> ModelInputBatch:
    """Build a featurized batch over every measured pair in *dataset*.

    Features are drawn rather than computed, so the batch says nothing about any
    featurizer - which is the point when the predictor is what is under test. The
    response is a noisy linear function of the drawn features, so a predictor
    that trains at all can beat the mean.

    Args:
        dataset: Dataset supplying the entity ids and the measured pairs.
        sample_block_names: Names to expose the sample matrix under, for
            predictors that read named blocks rather than a flat matrix.
        drug_block_names: Names to expose the drug matrix under.
        n_sample_features: Width of the drawn sample feature matrix.
        n_drug_features: Width of the drawn drug feature matrix, or ``None`` for
            a sample-only batch.
        seed: Seed for the drawn features and noise.

    Returns:
        A batch ready for :meth:`~drevalpy.plugin.Predictor.fit`.
    """
    rng = np.random.default_rng(seed)
    response = observed_pairs(dataset)
    sample_entity_ids = np.asarray(dataset.sample_ids)
    drug_entity_ids = np.asarray(dataset.drug_ids)

    sample_features = rng.normal(size=(len(sample_entity_ids), n_sample_features)).astype(np.float32)
    sample_pair_idx = _row_index(sample_entity_ids, response.sample_ids)

    drug_features = None
    drug_pair_idx = None
    if n_drug_features is not None:
        drug_features = rng.normal(size=(len(drug_entity_ids), n_drug_features)).astype(np.float32)
        drug_pair_idx = _row_index(drug_entity_ids, response.drug_ids)

    sample_blocks = {"features": numeric_feature_block(sample_features)}
    sample_blocks.update({name: numeric_feature_block(sample_features) for name in sample_block_names})
    sample_result = FeaturizerResult(sample_entity_ids, sample_blocks, ("features",))
    drug_result = None
    if drug_features is not None:
        drug_blocks_map = {"features": numeric_feature_block(drug_features)}
        drug_blocks_map.update({name: numeric_feature_block(drug_features) for name in drug_block_names})
        drug_result = FeaturizerResult(drug_entity_ids, drug_blocks_map, ("features",))

    return ModelInputBatch.from_response(
        _learnable_response(response, sample_features, sample_pair_idx, rng),
        sample=sample_result,
        drug=drug_result,
        sample_pair_idx=sample_pair_idx,
        drug_pair_idx=drug_pair_idx,
    )


def _row_index(entity_ids: np.ndarray, pair_ids: np.ndarray) -> np.ndarray:
    """Map each pair identifier onto its row in the feature matrix."""
    rows = {str(entity_id): row for row, entity_id in enumerate(entity_ids)}
    return np.asarray([rows[str(pair_id)] for pair_id in pair_ids], dtype=np.int64)


def _learnable_response(
    response: ResponseBatch,
    sample_features: np.ndarray,
    sample_pair_idx: np.ndarray,
    rng: np.random.Generator,
) -> ResponseBatch:
    """Replace the response with a noisy linear function of the drawn features.

    Without this the drawn features carry no signal, and a check asserting that
    a predictor learned something could only ever assert that it ran.
    """
    weights = rng.normal(size=sample_features.shape[1])
    signal = sample_features[sample_pair_idx] @ weights
    return ResponseBatch(
        response=(signal + rng.normal(scale=0.1, size=len(response))).astype(np.float64),
        sample_ids=response.sample_ids,
        drug_ids=response.drug_ids,
    )
