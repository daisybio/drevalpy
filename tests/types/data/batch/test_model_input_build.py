"""Tests for canonical entity-result batch construction."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_build import build_model_input_batch
from drevalpy.types.data.batch.response_batch import ResponseBatch


def _response(cell=("cl1", "cl2"), drug=("d1", "d2")) -> ResponseBatch:
    return ResponseBatch(np.arange(len(cell), dtype=float), np.asarray(cell), np.asarray(drug))


def _result(ids=("cl1", "cl2"), values=None) -> FeaturizerResult:
    matrix = np.arange(len(ids) * 2, dtype=np.float32).reshape(len(ids), 2) if values is None else values
    return FeaturizerResult.from_blocks({"view": numeric_feature_block(matrix)}, np.asarray(ids))


def test_build_indexes_entities_in_result_order() -> None:
    batch = build_model_input_batch(
        _response(("cl2", "cl1", "cl2"), ("d2", "d1", "d2")),
        sample=_result(),
        drug=_result(("d1", "d2")),
    )
    assert batch.sample_pair_idx.tolist() == [1, 0, 1]
    assert batch.drug_pair_idx.tolist() == [1, 0, 1]
    assert batch.sample.entity_ids.tolist() == ["cl1", "cl2"]


def test_drug_is_optional() -> None:
    batch = build_model_input_batch(_response(), sample=_result())
    assert batch.drug is None
    assert batch.drug_pair_idx is None


def test_featureless_entity_table_retains_and_indexes_entity_ids() -> None:
    featureless = FeaturizerResult(np.array(["cl1", "cl2"]), {}, ())
    batch = build_model_input_batch(_response(), sample=featureless)
    assert batch.sample_pair_idx.tolist() == [0, 1]


def test_build_rejects_duplicate_entity_ids() -> None:
    duplicate = FeaturizerResult(np.array(["cl1", "cl1"]), {}, ())
    with pytest.raises(ValueError, match="unique"):
        build_model_input_batch(_response(), sample=duplicate)


def test_zero_pairs_yield_empty_indices() -> None:
    empty = FeaturizerResult(np.array([], dtype=str), {}, ())
    batch = build_model_input_batch(_response((), ()), sample=empty, drug=empty)
    assert batch.sample_pair_idx.size == 0
    assert batch.drug_pair_idx is not None and batch.drug_pair_idx.size == 0


def test_result_rejects_mismatched_rows_and_scalars() -> None:
    with pytest.raises(ValueError, match="rows must match entity_ids"):
        _result(("cl1",), np.ones((2, 1)))
    with pytest.raises(ValueError, match="rows must match entity_ids"):
        _result(("cl1",), np.array(1.0))


def test_build_rejects_missing_pair_ids() -> None:
    with pytest.raises(ValueError, match="Missing sample identifiers"):
        build_model_input_batch(_response(("missing",), ("d1",)), sample=_result(), drug=_result(("d1", "d2")))
    with pytest.raises(ValueError, match="Missing drug identifiers"):
        build_model_input_batch(_response(("cl1",), ("missing",)), sample=_result(), drug=_result(("d1", "d2")))
