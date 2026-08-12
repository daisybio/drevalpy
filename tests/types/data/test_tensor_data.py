"""Tests for the lazy pair-level DataLoader factory.

``IndexedPairDataset`` exists to avoid materializing a pair-level feature matrix,
so the tests assert that a pair index reads through to the compact entity matrix
rather than checking a pre-expanded array.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drevalpy.types.data.tensor_data import IndexedPairDataset, make_pair_loader

CELL_LINE_FEATURES = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
DRUG_FEATURES = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
CELL_LINE_PAIR_IDX = np.array([0, 1, 0, 1])
DRUG_PAIR_IDX = np.array([0, 1, 2, 0])
RESPONSE = np.array([1.0, 2.0, 3.0, 4.0])


class TestIndexedPairDataset:
    def test_length_follows_the_pair_index(self):
        dataset = IndexedPairDataset((CELL_LINE_FEATURES, CELL_LINE_PAIR_IDX))

        assert len(dataset) == 4

    def test_length_is_zero_without_any_feature_spec(self):
        assert len(IndexedPairDataset()) == 0

    def test_getitem_reads_through_to_the_entity_row(self):
        dataset = IndexedPairDataset((CELL_LINE_FEATURES, CELL_LINE_PAIR_IDX))

        (features,) = dataset[1]

        torch.testing.assert_close(features, torch.tensor([0.3, 0.4]))

    def test_repeated_pair_indices_share_the_same_entity_row(self):
        dataset = IndexedPairDataset((CELL_LINE_FEATURES, CELL_LINE_PAIR_IDX))

        torch.testing.assert_close(dataset[0][0], dataset[2][0])

    def test_getitem_returns_one_tensor_per_feature_spec(self):
        dataset = IndexedPairDataset(
            (CELL_LINE_FEATURES, CELL_LINE_PAIR_IDX),
            (DRUG_FEATURES, DRUG_PAIR_IDX),
        )

        assert len(dataset[0]) == 2

    def test_response_is_appended_as_a_trailing_scalar(self):
        dataset = IndexedPairDataset((CELL_LINE_FEATURES, CELL_LINE_PAIR_IDX), response=RESPONSE)

        item = dataset[2]

        assert len(item) == 2
        torch.testing.assert_close(item[-1], torch.tensor(3.0))

    def test_features_are_cast_to_float32(self):
        dataset = IndexedPairDataset((CELL_LINE_FEATURES.astype(np.float64), CELL_LINE_PAIR_IDX))

        assert dataset[0][0].dtype is torch.float32

    def test_response_is_cast_to_float32(self):
        dataset = IndexedPairDataset((CELL_LINE_FEATURES, CELL_LINE_PAIR_IDX), response=RESPONSE)

        assert dataset[0][-1].dtype is torch.float32

    def test_pair_count_follows_the_first_feature_spec(self):
        """``__len__`` reads the first spec's index, so callers must keep specs aligned."""
        dataset = IndexedPairDataset(
            (CELL_LINE_FEATURES, CELL_LINE_PAIR_IDX),
            (DRUG_FEATURES, DRUG_PAIR_IDX[:2]),
        )

        assert len(dataset) == len(CELL_LINE_PAIR_IDX)

    def test_reading_past_a_shorter_spec_fails_loudly(self):
        """``strict=True`` zipping stops a silently truncated batch."""
        dataset = IndexedPairDataset(
            (CELL_LINE_FEATURES, CELL_LINE_PAIR_IDX),
            (DRUG_FEATURES, DRUG_PAIR_IDX[:2]),
        )

        with pytest.raises(IndexError):
            dataset[3]


class TestMakePairLoader:
    def test_loader_batches_pairs(self):
        loader = make_pair_loader(
            (CELL_LINE_FEATURES, CELL_LINE_PAIR_IDX),
            response=RESPONSE,
            batch_size=2,
            shuffle=False,
        )

        batches = list(loader)

        assert len(batches) == 2

    def test_batch_shapes_stack_the_entity_rows(self):
        loader = make_pair_loader(
            (CELL_LINE_FEATURES, CELL_LINE_PAIR_IDX),
            (DRUG_FEATURES, DRUG_PAIR_IDX),
            response=RESPONSE,
            batch_size=4,
            shuffle=False,
        )

        cell_lines, drugs, response = next(iter(loader))

        assert cell_lines.shape == (4, 2)
        assert drugs.shape == (4, 1)
        assert response.shape == (4,)

    def test_unshuffled_loader_preserves_pair_order(self):
        loader = make_pair_loader(
            (CELL_LINE_FEATURES, CELL_LINE_PAIR_IDX),
            response=RESPONSE,
            batch_size=4,
            shuffle=False,
        )

        _, response = next(iter(loader))

        torch.testing.assert_close(response, torch.tensor(RESPONSE, dtype=torch.float32))

    def test_drop_last_discards_an_incomplete_batch(self):
        loader = make_pair_loader(
            (CELL_LINE_FEATURES, CELL_LINE_PAIR_IDX),
            response=RESPONSE,
            batch_size=3,
            shuffle=False,
            drop_last=True,
        )

        assert [batch[-1].shape[0] for batch in loader] == [3]

    def test_keeping_the_last_batch_yields_every_pair(self):
        loader = make_pair_loader(
            (CELL_LINE_FEATURES, CELL_LINE_PAIR_IDX),
            response=RESPONSE,
            batch_size=3,
            shuffle=False,
            drop_last=False,
        )

        assert sum(batch[-1].shape[0] for batch in loader) == len(RESPONSE)

    def test_loader_without_a_response_yields_features_only(self):
        loader = make_pair_loader(
            (CELL_LINE_FEATURES, CELL_LINE_PAIR_IDX),
            batch_size=4,
            shuffle=False,
        )

        batch = next(iter(loader))

        assert len(batch) == 1
