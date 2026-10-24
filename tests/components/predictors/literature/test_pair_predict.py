"""Tests for the shared eval-time pair inference helper."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.literature._pair_predict import (
    PairEvalSpec,
    concatenated_forward,
    predict_pairs,
    require_drug_pair_idx,
)
from drevalpy.types.data.batch.feature_block import numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch
from drevalpy.types.data.batch.response_batch import ResponseBatch
from tests.components.predictors.literature._helpers import two_by_two_batch

pytestmark = pytest.mark.slow

SAMPLE_VALUES = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
DRUG_VALUES = np.array([[10.0], [20.0]], dtype=np.float32)


def _batch(*, with_drug_pair_idx: bool = True) -> ModelInputBatch:
    return two_by_two_batch(
        sample_blocks={"view": numeric_feature_block(SAMPLE_VALUES)},
        drug_blocks={"view": numeric_feature_block(DRUG_VALUES)},
        drug_pair_idx=np.array([0, 1, 0, 1]) if with_drug_pair_idx else None,
    )


def _spec(batch_size: int = 2):
    import torch

    return PairEvalSpec(
        sample_blocks=(SAMPLE_VALUES,),
        drug_blocks=(DRUG_VALUES,),
        batch_size=batch_size,
        device=torch.device("cpu"),
    )


class _SumModel:
    def __init__(self) -> None:
        self.training = True
        self.seen_shapes: list[tuple[int, ...]] = []

    def eval(self) -> None:
        self.training = False

    def __call__(self, *tensors):
        import torch

        self.seen_shapes.append(tuple(tensors[0].shape))
        return torch.stack([tensor.sum(dim=1) for tensor in tensors]).sum(dim=0)


class _WidthModel:
    def eval(self) -> None:
        return None

    def __call__(self, tensor):
        import torch

        return torch.full((tensor.shape[0],), float(tensor.shape[1]))


def test_require_drug_pair_idx_passes_arrays_and_rejects_none() -> None:
    indices = np.array([0, 1])
    assert require_drug_pair_idx(indices) is indices
    with pytest.raises(RuntimeError, match="drug_pair_idx is required for this predictor"):
        require_drug_pair_idx(None)


def test_predict_pairs_returns_float64_in_pair_order_and_switches_to_eval() -> None:
    model = _SumModel()

    predictions = predict_pairs(model, _batch(), _spec())

    np.testing.assert_allclose(predictions, [13.0, 23.0, 17.0, 27.0])
    assert predictions.dtype == np.float64
    assert model.training is False


def test_a_batch_size_below_the_pair_count_still_covers_every_pair() -> None:
    model = _SumModel()

    assert len(predict_pairs(model, _batch(), _spec(batch_size=3))) == 4
    assert [shape[0] for shape in model.seen_shapes] == [3, 1]


def test_a_batch_without_drug_pair_indices_is_rejected() -> None:
    with pytest.raises(ValueError, match="drug feature result requires pair indices"):
        _batch(with_drug_pair_idx=False)


def test_a_multi_element_output_row_is_flattened() -> None:
    class _ColumnModel(_SumModel):
        def __call__(self, *tensors):
            return super().__call__(*tensors).reshape(-1, 1)

    assert predict_pairs(_ColumnModel(), _batch(), _spec()).shape == (4,)


def test_a_batch_with_no_pairs_yields_an_empty_float64_array() -> None:
    empty = two_by_two_batch(
        response=ResponseBatch(
            response=np.empty(0), sample_ids=np.empty(0, dtype=str), drug_ids=np.empty(0, dtype=str)
        ),
        sample_blocks={"view": numeric_feature_block(SAMPLE_VALUES)},
        drug_blocks={"view": numeric_feature_block(DRUG_VALUES)},
        sample_pair_idx=np.empty(0, dtype=np.intp),
        drug_pair_idx=np.empty(0, dtype=np.intp),
    )

    predictions = predict_pairs(_SumModel(), empty, _spec())

    assert predictions.shape == (0,)
    assert predictions.dtype == np.float64


def test_concatenated_forward_joins_blocks_feature_wise_otherwise_they_stay_separate() -> None:
    predictions = predict_pairs(_WidthModel(), _batch(), _spec(), forward=concatenated_forward(_WidthModel()))
    np.testing.assert_allclose(predictions, np.full(4, 3.0))

    model = _SumModel()
    predict_pairs(model, _batch(), _spec(batch_size=4))
    assert model.seen_shapes == [(4, 2)]
