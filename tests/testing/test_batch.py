"""Tests for :mod:`drevalpy.testing.batch`."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.testing.batch import (
    N_CELL_LINE_FEATURES,
    N_DRUG_FEATURES,
    build_synthetic_batch,
    observed_pairs,
)
from drevalpy.testing.synthetic import build_synthetic_dataset
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


@pytest.fixture(scope="module")
def dataset():
    return build_synthetic_dataset()


@pytest.fixture(scope="module")
def batch(dataset) -> ModelInputBatch:
    return build_synthetic_batch(dataset)


class TestObservedPairs:
    def test_one_triple_per_measured_entry(self, dataset):
        pairs = observed_pairs(dataset)

        assert len(pairs) == int((~np.isnan(dataset.response_matrix)).sum())

    def test_no_unmeasured_pair_is_included(self, dataset):
        pairs = observed_pairs(dataset)

        assert np.isfinite(pairs.response).all()

    def test_ids_are_drawn_from_the_dataset(self, dataset):
        pairs = observed_pairs(dataset)

        assert set(pairs.cell_line_ids) <= set(dataset.cell_line_ids)
        assert set(pairs.drug_ids) <= set(dataset.drug_ids)

    def test_all_three_arrays_are_the_same_length(self, dataset):
        pairs = observed_pairs(dataset)

        assert len(pairs.cell_line_ids) == len(pairs.drug_ids) == len(pairs.response)


class TestBatchShape:
    def test_it_returns_a_model_input_batch(self, batch):
        assert isinstance(batch, ModelInputBatch)

    def test_there_is_one_pair_per_measured_entry(self, batch, dataset):
        assert batch.n_pairs == int((~np.isnan(dataset.response_matrix)).sum())

    def test_feature_matrices_have_the_default_widths(self, batch):
        assert batch.cell_line_features.shape[1] == N_CELL_LINE_FEATURES
        assert batch.drug_features.shape[1] == N_DRUG_FEATURES

    def test_feature_widths_are_configurable(self, dataset):
        narrow = build_synthetic_batch(dataset, n_cell_line_features=3, n_drug_features=2)

        assert narrow.cell_line_features.shape[1] == 3
        assert narrow.drug_features.shape[1] == 2

    def test_feature_rows_are_per_entity_not_per_pair(self, batch, dataset):
        assert batch.cell_line_features.shape[0] == len(dataset.cell_line_ids)
        assert batch.drug_features.shape[0] == len(dataset.drug_ids)

    def test_pair_indices_address_the_feature_rows(self, batch):
        assert batch.cell_line_pair_idx.max() < batch.cell_line_features.shape[0]
        assert batch.drug_pair_idx.max() < batch.drug_features.shape[0]

    def test_the_design_matrix_is_pair_aligned(self, batch):
        matrix = batch.to_feature_matrix()

        assert matrix.shape == (batch.n_pairs, N_CELL_LINE_FEATURES + N_DRUG_FEATURES)

    def test_features_are_finite(self, batch):
        assert np.isfinite(batch.cell_line_features).all()
        assert np.isfinite(batch.drug_features).all()


class TestDrugFreeBatch:
    def test_no_drug_features_are_produced(self, dataset):
        cell_line_only = build_synthetic_batch(dataset, n_drug_features=None)

        assert cell_line_only.drug_features is None
        assert cell_line_only.drug_pair_idx is None

    def test_the_design_matrix_covers_the_cell_line_side_only(self, dataset):
        cell_line_only = build_synthetic_batch(dataset, n_drug_features=None)

        assert cell_line_only.to_feature_matrix().shape[1] == N_CELL_LINE_FEATURES


class TestNamedBlocks:
    def test_no_blocks_are_exposed_by_default(self, batch):
        assert batch.cell_line_blocks == {}
        assert batch.drug_blocks == {}

    def test_requested_blocks_hold_the_feature_matrix(self, dataset):
        with_blocks = build_synthetic_batch(
            dataset,
            cell_line_block_names=["identity"],
            drug_block_names=["fingerprints"],
        )

        np.testing.assert_array_equal(
            with_blocks.cell_line_blocks["identity"].values,
            with_blocks.cell_line_features,
        )
        np.testing.assert_array_equal(
            with_blocks.drug_blocks["fingerprints"].values,
            with_blocks.drug_features,
        )

    def test_drug_blocks_are_skipped_when_there_are_no_drug_features(self, dataset):
        with_blocks = build_synthetic_batch(dataset, drug_block_names=["x"], n_drug_features=None)

        assert with_blocks.drug_blocks == {}


class TestLearnableResponse:
    def test_the_response_is_finite(self, batch):
        assert np.isfinite(batch.response).all()

    def test_it_carries_signal_a_linear_model_can_recover(self, batch):
        """Otherwise ``check_predictor_fit_predict`` could only assert that it ran."""
        from sklearn.linear_model import Ridge

        matrix = batch.to_feature_matrix()
        model = Ridge().fit(matrix, batch.response)

        residual = np.mean((model.predict(matrix) - batch.response) ** 2)
        assert residual < np.var(batch.response) / 10

    def test_it_replaces_the_dataset_response(self, batch, dataset):
        """The drawn features carry the signal, so the raw responses cannot."""
        assert not np.allclose(batch.response, observed_pairs(dataset).response)


class TestDeterminism:
    def test_the_same_seed_gives_the_same_features(self, dataset):
        first = build_synthetic_batch(dataset)
        second = build_synthetic_batch(dataset)

        np.testing.assert_array_equal(first.cell_line_features, second.cell_line_features)
        np.testing.assert_array_equal(first.response, second.response)

    def test_a_different_seed_gives_different_features(self, dataset):
        first = build_synthetic_batch(dataset, seed=1)
        second = build_synthetic_batch(dataset, seed=2)

        assert not np.array_equal(first.cell_line_features, second.cell_line_features)
