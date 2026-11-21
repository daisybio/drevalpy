"""Synthetic :class:`PredictorInput` construction for predictor tests.

Predictors consume a fully featurized :class:`PredictorInput`, so exercising one normally means
composing a whole model. This module short-circuits that: it draws dense feature
matrices directly and pairs them with the measured entries of a dataset's
response matrix, so a predictor can be trained without any featurizer at all.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import replace

import numpy as np

from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.feature_block import FeatureBlock, FeaturizerOutput
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.data.response_table import ResponseTable

N_SAMPLE_FEATURES = 8
N_DRUG_FEATURES = 4
SEED = 20260813


def observed_pairs(dataset: Dataset, *, layers: Collection[str] = ()) -> ResponseTable:
    """Return every measured sample/drug pair in *dataset*.

    :param dataset: Dataset whose response matrix is read.
    :param layers: Response layers to read at the same pairs; names the dataset does not have are left out.
    :returns: Response pairs for the non-NaN entries, in row-major order.
    """
    matrix = dataset.response_matrix
    rows, columns = np.nonzero(~np.isnan(matrix))
    available = set(dataset.measures())
    return ResponseTable(
        response=matrix[rows, columns].astype(np.float64),
        sample_ids=np.asarray(dataset.sample_ids)[rows],
        drug_ids=np.asarray(dataset.drug_ids)[columns],
        layers={name: dataset.get_response_layer(name)[rows, columns] for name in layers if name in available},
    )


def build_synthetic_input(
    dataset: Dataset,
    *,
    sample_block_names: Sequence[str] = (),
    drug_block_names: Sequence[str] = (),
    n_sample_features: int = N_SAMPLE_FEATURES,
    n_drug_features: int | None = N_DRUG_FEATURES,
    seed: int = SEED,
    response_layers: Collection[str] = (),
) -> PredictorInput:
    """Build a featurized :class:`PredictorInput` over every measured pair in *dataset*.

    Features are drawn rather than computed, so the input says nothing about any
    featurizer - which is the point when the predictor is what is under test. The
    response is a noisy linear function of the drawn features, so a predictor
    that trains at all can beat the mean.

    :param dataset: Dataset supplying the entity ids and the measured pairs.
    :param sample_block_names: Names to expose the sample matrix under, for
        predictors that read named blocks rather than a flat matrix.
    :param drug_block_names: Names to expose the drug matrix under.
    :param n_sample_features: Width of the drawn sample feature matrix.
    :param n_drug_features: Width of the drawn drug feature matrix, or ``None`` for
        a sample-only input.
    :param seed: Seed for the drawn features and noise.
    :param response_layers: Response layers of *dataset* to carry in ``data.response_layers``, as
        ``Predictor.response_layers`` declares them; missing ones are left out.
    :returns: A :class:`PredictorInput` ready for :meth:`~drevalpy.plugin.Predictor.fit`.
    """
    rng = np.random.default_rng(seed)
    response = observed_pairs(dataset, layers=response_layers)
    sample_entity_ids = np.asarray(dataset.sample_ids)
    drug_entity_ids = np.asarray(dataset.drug_ids)

    sample_features = rng.normal(size=(len(sample_entity_ids), n_sample_features)).astype(np.float32)

    drug_features = None
    if n_drug_features is not None:
        drug_features = rng.normal(size=(len(drug_entity_ids), n_drug_features)).astype(np.float32)

    sample_names = tuple(sample_block_names) if sample_block_names else ("features",)
    sample_blocks = {name: FeatureBlock(sample_features) for name in sample_names}
    sample_result = FeaturizerOutput(sample_entity_ids, sample_blocks)
    drug_result = None
    if drug_features is not None:
        drug_names = tuple(drug_block_names) if drug_block_names else ("features",)
        drug_blocks = {name: FeatureBlock(drug_features) for name in drug_names}
        drug_result = FeaturizerOutput(drug_entity_ids, drug_blocks)

    return PredictorInput(
        _learnable_response(response, sample_features, sample_result.rows_for(response.sample_ids), rng),
        sample=sample_result,
        drug=drug_result,
    )


def _learnable_response(
    response: ResponseTable,
    sample_features: np.ndarray,
    sample_pair_idx: np.ndarray,
    rng: np.random.Generator,
) -> ResponseTable:
    """Replace the response with a noisy linear function of the drawn features.

    Without this the drawn features carry no signal, and a check asserting that
    a predictor learned something could only ever assert that it ran.
    """
    weights = rng.normal(size=sample_features.shape[1])
    signal = sample_features[sample_pair_idx] @ weights
    return replace(response, response=(signal + rng.normal(scale=0.1, size=len(response))).astype(np.float64))
