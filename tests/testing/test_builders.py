"""Tests for the shipped synthetic input builder ``drevalpy.testing.inputs``."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from drevalpy.testing.inputs import N_DRUG_FEATURES, N_SAMPLE_FEATURES, build_synthetic_input, observed_pairs


@pytest.fixture(scope="module")
def model_input(synthetic_dataset):
    return build_synthetic_input(synthetic_dataset)


def test_observed_pairs_cover_exactly_the_measured_entries(synthetic_dataset):
    pairs = observed_pairs(synthetic_dataset)
    assert (
        len(pairs.sample_ids)
        == len(pairs.drug_ids)
        == len(pairs.response)
        == int((~np.isnan(synthetic_dataset.response_matrix)).sum())
    )
    assert np.isfinite(pairs.response).all()
    assert set(pairs.sample_ids) <= set(synthetic_dataset.sample_ids)
    assert set(pairs.drug_ids) <= set(synthetic_dataset.drug_ids)


def test_input_shape(model_input, synthetic_dataset):
    assert len(model_input) == int((~np.isnan(synthetic_dataset.response_matrix)).sum())
    assert model_input.sample.matrix.shape == (len(synthetic_dataset.sample_ids), N_SAMPLE_FEATURES)
    assert model_input.drug.matrix.shape == (len(synthetic_dataset.drug_ids), N_DRUG_FEATURES)
    assert model_input.sample_pair_idx.max() < model_input.sample.matrix.shape[0]
    assert model_input.drug_pair_idx.max() < model_input.drug.matrix.shape[0]
    assert model_input.to_feature_matrix().shape == (len(model_input), N_SAMPLE_FEATURES + N_DRUG_FEATURES)
    assert np.isfinite(model_input.sample.matrix).all()
    assert np.isfinite(model_input.drug.matrix).all()
    assert set(model_input.sample.blocks) == set(model_input.drug.blocks) == {"features"}


def test_feature_widths_are_configurable(synthetic_dataset):
    narrow = build_synthetic_input(synthetic_dataset, n_sample_features=3, n_drug_features=2)
    assert (narrow.sample.matrix.shape[1], narrow.drug.matrix.shape[1]) == (3, 2)


def test_drug_free_input(synthetic_dataset):
    sample_only = build_synthetic_input(synthetic_dataset, n_drug_features=None, drug_block_names=["x"])
    assert sample_only.drug is None
    assert sample_only.drug_pair_idx is None
    assert sample_only.to_feature_matrix().shape[1] == N_SAMPLE_FEATURES


def test_requested_blocks_hold_the_feature_matrix(synthetic_dataset):
    with_blocks = build_synthetic_input(
        synthetic_dataset, sample_block_names=["oneHot"], drug_block_names=["fingerprints"]
    )
    np.testing.assert_array_equal(with_blocks.sample.blocks["oneHot"].values, with_blocks.sample.matrix)
    np.testing.assert_array_equal(with_blocks.drug.blocks["fingerprints"].values, with_blocks.drug.matrix)


def test_response_is_a_finite_signal_a_linear_model_can_recover(model_input, synthetic_dataset):
    assert np.isfinite(model_input.response).all()
    assert not np.allclose(model_input.response, observed_pairs(synthetic_dataset).response)
    matrix = model_input.to_feature_matrix()
    residual = np.mean((Ridge().fit(matrix, model_input.response).predict(matrix) - model_input.response) ** 2)
    assert residual < np.var(model_input.response) / 10


def test_seed_determines_the_features(synthetic_dataset):
    first, second = build_synthetic_input(synthetic_dataset), build_synthetic_input(synthetic_dataset)
    np.testing.assert_array_equal(first.sample.matrix, second.sample.matrix)
    np.testing.assert_array_equal(first.response, second.response)
    other = build_synthetic_input(synthetic_dataset, seed=2)
    assert not np.array_equal(first.sample.matrix, other.sample.matrix)


def test_an_input_carries_the_requested_response_layers(synthetic_dataset):
    data = build_synthetic_input(synthetic_dataset, response_layers=("regulation", "no_such_layer"))

    assert set(data.response_layers) == {"regulation"}
    pairs = observed_pairs(synthetic_dataset)
    rows = {sample_id: i for i, sample_id in enumerate(synthetic_dataset.sample_ids)}
    cols = {drug_id: j for j, drug_id in enumerate(synthetic_dataset.drug_ids)}
    expected = synthetic_dataset.get_response_layer("regulation")[
        [rows[s] for s in pairs.sample_ids], [cols[d] for d in pairs.drug_ids]
    ]
    np.testing.assert_array_equal(data.response_layers["regulation"], expected)
