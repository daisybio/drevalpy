"""The torch loader over pair-indexed feature arrays and the eval pass over it."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drevalpy.components.predictors._pair_loader import (
    IndexedPairDataset,
    concatenated_forward,
    make_pair_loader,
    pair_loader,
    predict_pairs,
    require_drug_pair_idx,
)
from drevalpy.types.data.feature_block import FeatureBlock
from drevalpy.types.data.predictor_input import PredictorInput
from drevalpy.types.data.response_table import ResponseTable
from tests.components.predictors.literature._helpers import early_stopping, two_by_two_batch

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


def _two_by_two(*, with_early_stopping: bool = True):
    return two_by_two_batch(
        sample_blocks={"view": FeatureBlock(SAMPLE_FEATURES)},
        drug_blocks={"view": FeatureBlock(DRUG_FEATURES[:2])},
        early_stopping=early_stopping() if with_early_stopping else None,
    )


class TestPairLoader:
    def test_training_pairs_carry_the_float32_response(self):
        loader = pair_loader(_two_by_two(), SAMPLE_FEATURES, DRUG_FEATURES[:2], batch_size=4, shuffle=True)

        samples, drugs, response = next(iter(loader))

        assert isinstance(loader.sampler, torch.utils.data.RandomSampler)
        assert (samples.shape, drugs.shape) == ((4, 2), (4, 1))
        assert response.dtype is torch.float32
        assert sorted(response.tolist()) == [1.0, 2.0, 3.0, 4.0]

    def test_the_stopping_pairs_read_the_same_entity_rows_with_their_own_response(self):
        stopping = _two_by_two().stopping
        assert stopping is not None
        loader = pair_loader(stopping, SAMPLE_FEATURES, DRUG_FEATURES[:2], batch_size=4, shuffle=False)

        samples, drugs, response = next(iter(loader))

        assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
        torch.testing.assert_close(response, torch.tensor([1.5, 2.5]))
        torch.testing.assert_close(samples, torch.tensor(SAMPLE_FEATURES))
        torch.testing.assert_close(drugs, torch.tensor(DRUG_FEATURES[:2]))

    def test_drop_last_is_passed_on(self):
        loader = pair_loader(
            _two_by_two(), SAMPLE_FEATURES, DRUG_FEATURES[:2], batch_size=3, shuffle=False, drop_last=True
        )

        assert [batch[0].shape[0] for batch in loader] == [3]

    def test_a_side_given_as_none_and_the_response_can_be_left_out(self):
        loader = pair_loader(_two_by_two(), None, DRUG_FEATURES[:2], batch_size=4, shuffle=False, response=False)

        (drugs,) = next(iter(loader))

        torch.testing.assert_close(drugs, torch.tensor(DRUG_FEATURES[[0, 1, 0, 1]]))

    def test_a_drug_matrix_needs_drug_pair_indices(self):
        batch = _two_by_two()
        featureless = PredictorInput(batch.pairs, sample=batch.sample)

        with pytest.raises(RuntimeError, match="drug_pair_idx is required"):
            pair_loader(featureless, SAMPLE_FEATURES, DRUG_FEATURES[:2], batch_size=2, shuffle=False)


def test_require_drug_pair_idx_passes_arrays_and_rejects_none():
    indices = np.array([0, 1])
    assert require_drug_pair_idx(indices) is indices
    with pytest.raises(RuntimeError, match="drug_pair_idx is required for this predictor"):
        require_drug_pair_idx(None)


PREDICT_SAMPLE_VALUES = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
PREDICT_DRUG_VALUES = np.array([[10.0], [20.0]], dtype=np.float32)


def _predict_batch() -> PredictorInput:
    return two_by_two_batch(
        sample_blocks={"view": FeatureBlock(PREDICT_SAMPLE_VALUES)},
        drug_blocks={"view": FeatureBlock(PREDICT_DRUG_VALUES)},
    )


def _predict(model, batch, *, batch_size: int = 2, forward=None) -> np.ndarray:
    return predict_pairs(
        model,
        batch,
        PREDICT_SAMPLE_VALUES,
        PREDICT_DRUG_VALUES,
        batch_size=batch_size,
        device=torch.device("cpu"),
        forward=forward,
    )


class _SumModel:
    def __init__(self) -> None:
        self.training = True
        self.seen_shapes: list[tuple[int, ...]] = []

    def eval(self) -> None:
        self.training = False

    def __call__(self, *tensors):
        self.seen_shapes.append(tuple(tensors[0].shape))
        return torch.stack([tensor.sum(dim=1) for tensor in tensors]).sum(dim=0)


class _WidthModel:
    def eval(self) -> None:
        return None

    def __call__(self, tensor):
        return torch.full((tensor.shape[0],), float(tensor.shape[1]))


def test_predict_pairs_returns_float64_in_pair_order_and_switches_to_eval() -> None:
    model = _SumModel()

    predictions = _predict(model, _predict_batch())

    np.testing.assert_allclose(predictions, [13.0, 23.0, 17.0, 27.0])
    assert predictions.dtype == np.float64
    assert model.training is False


def test_a_batch_size_below_the_pair_count_still_covers_every_pair() -> None:
    model = _SumModel()

    assert len(_predict(model, _predict_batch(), batch_size=3)) == 4
    assert [shape[0] for shape in model.seen_shapes] == [3, 1]


def test_a_multi_element_output_row_is_flattened() -> None:
    class _ColumnModel(_SumModel):
        def __call__(self, *tensors):
            return super().__call__(*tensors).reshape(-1, 1)

    assert _predict(_ColumnModel(), _predict_batch()).shape == (4,)


def test_a_batch_with_no_pairs_yields_an_empty_float64_array() -> None:
    empty = two_by_two_batch(
        response=ResponseTable(
            response=np.empty(0), sample_ids=np.empty(0, dtype=str), drug_ids=np.empty(0, dtype=str)
        ),
        sample_blocks={"view": FeatureBlock(PREDICT_SAMPLE_VALUES)},
        drug_blocks={"view": FeatureBlock(PREDICT_DRUG_VALUES)},
    )

    predictions = _predict(_SumModel(), empty)

    assert predictions.shape == (0,)
    assert predictions.dtype == np.float64


def test_concatenated_forward_joins_blocks_feature_wise_otherwise_they_stay_separate() -> None:
    predictions = _predict(_WidthModel(), _predict_batch(), forward=concatenated_forward(_WidthModel()))
    np.testing.assert_allclose(predictions, np.full(4, 3.0))

    model = _SumModel()
    _predict(model, _predict_batch(), batch_size=4)
    assert model.seen_shapes == [(4, 2)]
