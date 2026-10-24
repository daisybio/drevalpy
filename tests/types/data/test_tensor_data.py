"""Tests for the lazy pair-level DataLoader factory: pair indices read through to compact entity matrices."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drevalpy.types.data.tensor_data import IndexedPairDataset, make_pair_loader

SAMPLE_FEATURES = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
DRUG_FEATURES = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
SAMPLE_PAIR_IDX = np.array([0, 1, 0, 1])
DRUG_PAIR_IDX = np.array([0, 1, 2, 0])
RESPONSE = np.array([1.0, 2.0, 3.0, 4.0])


class TestIndexedPairDataset:
    def test_length(self):
        assert len(IndexedPairDataset((SAMPLE_FEATURES, SAMPLE_PAIR_IDX))) == 4
        assert len(IndexedPairDataset()) == 0

    def test_getitem_reads_through_to_the_entity_row(self):
        dataset = IndexedPairDataset((SAMPLE_FEATURES, SAMPLE_PAIR_IDX), (DRUG_FEATURES, DRUG_PAIR_IDX))

        assert len(dataset[0]) == 2
        torch.testing.assert_close(dataset[1][0], torch.tensor([0.3, 0.4]))
        torch.testing.assert_close(dataset[0][0], dataset[2][0])

    def test_response_is_appended_as_a_trailing_scalar(self):
        item = IndexedPairDataset((SAMPLE_FEATURES, SAMPLE_PAIR_IDX), response=RESPONSE)[2]

        assert len(item) == 2
        torch.testing.assert_close(item[-1], torch.tensor(3.0))
        assert item[-1].dtype is torch.float32

    def test_features_are_cast_to_float32(self):
        dataset = IndexedPairDataset((SAMPLE_FEATURES.astype(np.float64), SAMPLE_PAIR_IDX))

        assert dataset[0][0].dtype is torch.float32

    def test_pair_count_follows_the_first_spec_and_a_shorter_spec_fails_loudly(self):
        dataset = IndexedPairDataset((SAMPLE_FEATURES, SAMPLE_PAIR_IDX), (DRUG_FEATURES, DRUG_PAIR_IDX[:2]))

        assert len(dataset) == len(SAMPLE_PAIR_IDX)
        with pytest.raises(IndexError):
            dataset[3]


class TestMakePairLoader:
    def test_batches_stack_the_entity_rows_in_pair_order(self):
        loader = make_pair_loader(
            (SAMPLE_FEATURES, SAMPLE_PAIR_IDX),
            (DRUG_FEATURES, DRUG_PAIR_IDX),
            response=RESPONSE,
            batch_size=4,
            shuffle=False,
        )

        samples, drugs, response = next(iter(loader))

        assert samples.shape == (4, 2)
        assert drugs.shape == (4, 1)
        torch.testing.assert_close(response, torch.tensor(RESPONSE, dtype=torch.float32))

    @pytest.mark.parametrize(("drop_last", "sizes"), [(True, [3]), (False, [3, 1])])
    def test_drop_last(self, drop_last, sizes):
        loader = make_pair_loader(
            (SAMPLE_FEATURES, SAMPLE_PAIR_IDX), response=RESPONSE, batch_size=3, shuffle=False, drop_last=drop_last
        )

        assert [batch[-1].shape[0] for batch in loader] == sizes

    def test_loader_without_a_response_yields_features_only(self):
        loader = make_pair_loader((SAMPLE_FEATURES, SAMPLE_PAIR_IDX), batch_size=4, shuffle=False)

        assert len(next(iter(loader))) == 1
