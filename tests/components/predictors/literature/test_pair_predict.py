"""Tests for the shared eval-time pair inference helper.

``literature/_pair_predict.py`` replaced the ``_predict`` skeleton PharmaFormer,
Precily and SparseGO each carried: resolve the pair indices, refuse a batch without
``drug_pair_idx``, build a non-shuffled ``make_pair_loader``, and accumulate
predictions under ``torch.no_grad()``. What matters is that the output stays in pair
order, that the model is left in eval mode, and that ``concatenated_forward``
reproduces the two-block concatenation Precily and SparseGO do by hand.
"""

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

#: Every test below builds torch tensors through ``make_pair_loader``.
pytestmark = pytest.mark.slow

CELL_LINE_VALUES = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
DRUG_VALUES = np.array([[10.0], [20.0]], dtype=np.float32)


def _batch(*, with_drug_pair_idx: bool = True) -> ModelInputBatch:
    """Build a four-pair batch over two cell lines and two drugs.

    :param with_drug_pair_idx: Whether to populate ``drug_pair_idx``.
    :returns: Featurized ``ModelInputBatch``.
    """
    return two_by_two_batch(
        cell_line_blocks={"view": numeric_feature_block(CELL_LINE_VALUES)},
        drug_blocks={"view": numeric_feature_block(DRUG_VALUES)},
        drug_pair_idx=np.array([0, 1, 0, 1]) if with_drug_pair_idx else None,
    )


def _spec(batch_size: int = 2):
    """Build a spec over the two blocks of :func:`_batch`, pinned to the CPU.

    :param batch_size: Mini-batch size for the eval pass.
    :returns: A ``PairEvalSpec``.
    """
    import torch

    return PairEvalSpec(
        cell_line_blocks=(CELL_LINE_VALUES,),
        drug_blocks=(DRUG_VALUES,),
        batch_size=batch_size,
        device=torch.device("cpu"),
    )


class _SumModel:
    """Records eval-mode transitions and sums each block's features per row."""

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
    """Returns the feature width of the single tensor it is handed."""

    def eval(self) -> None:
        return None

    def __call__(self, tensor):
        import torch

        return torch.full((tensor.shape[0],), float(tensor.shape[1]))


class TestRequireDrugPairIdx:
    def test_an_array_passes_through_unchanged(self) -> None:
        indices = np.array([0, 1])

        assert require_drug_pair_idx(indices) is indices

    def test_none_is_the_documented_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="drug_pair_idx is required for this predictor"):
            require_drug_pair_idx(None)


class TestPredictPairs:
    def test_one_prediction_per_pair_in_pair_order(self) -> None:
        # Pairs are (cl0,d0), (cl0,d1), (cl1,d0), (cl1,d1); the model sums both blocks.
        predictions = predict_pairs(_SumModel(), _batch(), _spec())

        np.testing.assert_allclose(predictions, [13.0, 23.0, 17.0, 27.0])

    def test_the_result_is_float64_regardless_of_the_model_dtype(self) -> None:
        assert predict_pairs(_SumModel(), _batch(), _spec()).dtype == np.float64

    def test_the_model_is_switched_into_eval_mode(self) -> None:
        model = _SumModel()

        predict_pairs(model, _batch(), _spec())

        assert model.training is False

    def test_a_batch_size_below_the_pair_count_still_covers_every_pair(self) -> None:
        model = _SumModel()

        predictions = predict_pairs(model, _batch(), _spec(batch_size=3))

        assert len(predictions) == 4
        # 4 pairs at batch_size 3: no drop_last, so the trailing single pair survives.
        assert [shape[0] for shape in model.seen_shapes] == [3, 1]

    def test_a_batch_without_drug_pair_indices_is_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="drug_pair_idx is required"):
            predict_pairs(_SumModel(), _batch(with_drug_pair_idx=False), _spec())

    def test_a_multi_element_output_row_is_flattened(self) -> None:
        """A model returning ``[batch, 1]`` must not yield a nested result."""

        class _ColumnModel(_SumModel):
            def __call__(self, *tensors):
                return super().__call__(*tensors).reshape(-1, 1)

        assert predict_pairs(_ColumnModel(), _batch(), _spec()).shape == (4,)

    def test_a_batch_with_no_pairs_yields_an_empty_float64_array(self) -> None:
        """``np.concatenate`` rejects an empty list, so the no-batch case is separate."""
        empty = two_by_two_batch(
            response=ResponseBatch(
                response=np.empty(0),
                cell_line_ids=np.empty(0, dtype=str),
                drug_ids=np.empty(0, dtype=str),
            ),
            cell_line_blocks={"view": numeric_feature_block(CELL_LINE_VALUES)},
            drug_blocks={"view": numeric_feature_block(DRUG_VALUES)},
            cell_line_pair_idx=np.empty(0, dtype=np.intp),
            drug_pair_idx=np.empty(0, dtype=np.intp),
        )

        predictions = predict_pairs(_SumModel(), empty, _spec())

        assert predictions.shape == (0,)
        assert predictions.dtype == np.float64


class TestConcatenatedForward:
    def test_the_blocks_arrive_concatenated_feature_wise(self) -> None:
        predictions = predict_pairs(
            _WidthModel(),
            _batch(),
            _spec(),
            forward=concatenated_forward(_WidthModel()),
        )

        # 2 cell-line columns + 1 drug column.
        np.testing.assert_allclose(predictions, np.full(4, 3.0))

    def test_without_it_each_block_is_passed_separately(self) -> None:
        model = _SumModel()

        predict_pairs(model, _batch(), _spec(batch_size=4))

        assert model.seen_shapes == [(4, 2)]
