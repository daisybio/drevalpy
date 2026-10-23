"""Tests for the EnsembleMF batch helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drevalpy.components.predictors.literature.ensemblemf.data import drug_table, sorted_entities, split_pairs
from drevalpy.components.predictors.literature.ensemblemf.training import PairTensors
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch


def _pairs(n: int) -> PairTensors:
    return PairTensors(torch.arange(n), torch.arange(n) % 3, torch.arange(n, dtype=torch.float32))


def test_sorted_entities_orders_rows_by_id_and_maps_original_rows() -> None:
    ids = np.array(["b", "c", "a"])
    values = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    result = FeaturizerResult.from_blocks({"x": numeric_feature_block(values)}, ids)

    sorted_ids, matrix, position = sorted_entities(result, "x")

    assert sorted_ids.tolist() == ["a", "b", "c"]
    assert matrix[:, 0].tolist() == [3.0, 1.0, 2.0]
    assert position.tolist() == [1, 2, 0]


def test_drug_table_requires_drug_features() -> None:
    sample = FeaturizerResult.from_blocks({"x": numeric_feature_block(np.zeros((1, 1), dtype=np.float32))}, ["s"])
    batch = ModelInputBatch.from_response(
        ResponseBatch(response=np.zeros(1, dtype=np.float32), sample_ids=np.array(["s"]), drug_ids=np.array(["d"])),
        sample=sample,
        sample_pair_idx=np.zeros(1, dtype=np.int64),
    )

    with pytest.raises(RuntimeError, match="requires drug features"):
        drug_table(batch)


def test_split_pairs_keeps_the_supplied_validation_set() -> None:
    train, validation = _pairs(10), _pairs(3)

    kept_train, kept_validation = split_pairs(train, validation, 0)

    assert kept_train is train
    assert kept_validation is validation


def test_split_pairs_holds_out_a_seeded_tenth_without_overlap() -> None:
    train_a, validation_a = split_pairs(_pairs(30), None, 7)
    train_b, validation_b = split_pairs(_pairs(30), None, 7)

    assert len(validation_a) == 3
    assert len(train_a) == 27
    assert set(train_a.sample_idx.tolist()).isdisjoint(validation_a.sample_idx.tolist())
    assert train_a.sample_idx.tolist() == train_b.sample_idx.tolist()
    assert validation_a.sample_idx.tolist() == validation_b.sample_idx.tolist()


def test_split_pairs_accepts_an_unseeded_split() -> None:
    train, validation = split_pairs(_pairs(20), None, -1)

    assert len(train) + len(validation) == 20


def test_split_pairs_needs_two_pairs() -> None:
    with pytest.raises(ValueError, match="at least two training pairs"):
        split_pairs(_pairs(1), None, 0)
