"""Tests for batch."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from drevalpy.testing.batch import N_DRUG_FEATURES, N_SAMPLE_FEATURES, build_synthetic_batch, observed_pairs
from drevalpy.testing.synthetic import build_synthetic_dataset


@pytest.fixture(scope="module")
def dataset():
    return build_synthetic_dataset()


@pytest.fixture(scope="module")
def batch(dataset):
    return build_synthetic_batch(dataset)


def test_observed_pairs_cover_exactly_the_measured_entries(dataset):
    pairs = observed_pairs(dataset)
    assert (
        len(pairs.sample_ids)
        == len(pairs.drug_ids)
        == len(pairs.response)
        == int((~np.isnan(dataset.response_matrix)).sum())
    )
    assert np.isfinite(pairs.response).all()
    assert set(pairs.sample_ids) <= set(dataset.sample_ids)
    assert set(pairs.drug_ids) <= set(dataset.drug_ids)


def test_batch_shape(batch, dataset):
    assert batch.n_pairs == int((~np.isnan(dataset.response_matrix)).sum())
    assert batch.sample.matrix.shape == (len(dataset.sample_ids), N_SAMPLE_FEATURES)
    assert batch.drug.matrix.shape == (len(dataset.drug_ids), N_DRUG_FEATURES)
    assert batch.sample_pair_idx.max() < batch.sample.matrix.shape[0]
    assert batch.drug_pair_idx.max() < batch.drug.matrix.shape[0]
    assert batch.to_feature_matrix().shape == (batch.n_pairs, N_SAMPLE_FEATURES + N_DRUG_FEATURES)
    assert np.isfinite(batch.sample.matrix).all()
    assert np.isfinite(batch.drug.matrix).all()
    assert set(batch.sample.blocks) == set(batch.drug.blocks) == {"features"}


def test_feature_widths_are_configurable(dataset):
    narrow = build_synthetic_batch(dataset, n_sample_features=3, n_drug_features=2)
    assert (narrow.sample.matrix.shape[1], narrow.drug.matrix.shape[1]) == (3, 2)


def test_drug_free_batch(dataset):
    sample_only = build_synthetic_batch(dataset, n_drug_features=None, drug_block_names=["x"])
    assert sample_only.drug is None
    assert sample_only.drug_pair_idx is None
    assert sample_only.to_feature_matrix().shape[1] == N_SAMPLE_FEATURES


def test_requested_blocks_hold_the_feature_matrix(dataset):
    with_blocks = build_synthetic_batch(dataset, sample_block_names=["identity"], drug_block_names=["fingerprints"])
    np.testing.assert_array_equal(with_blocks.sample.blocks["identity"].values, with_blocks.sample.matrix)
    np.testing.assert_array_equal(with_blocks.drug.blocks["fingerprints"].values, with_blocks.drug.matrix)


def test_response_is_a_finite_signal_a_linear_model_can_recover(batch, dataset):
    assert np.isfinite(batch.response).all()
    assert not np.allclose(batch.response, observed_pairs(dataset).response)
    matrix = batch.to_feature_matrix()
    residual = np.mean((Ridge().fit(matrix, batch.response).predict(matrix) - batch.response) ** 2)
    assert residual < np.var(batch.response) / 10


def test_seed_determines_the_features(dataset):
    first, second = build_synthetic_batch(dataset), build_synthetic_batch(dataset)
    np.testing.assert_array_equal(first.sample.matrix, second.sample.matrix)
    np.testing.assert_array_equal(first.response, second.response)
    other = build_synthetic_batch(dataset, seed=2)
    assert not np.array_equal(first.sample.matrix, other.sample.matrix)
