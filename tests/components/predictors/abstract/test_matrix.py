"""Tests for the matrix predictor interface and the ``ModelInputBatch`` contract it relies on."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


def _ids(n: int) -> np.ndarray:
    return np.array([f"id_{i}" for i in range(n)])


def test_matrix_predictor_declares_the_matrix_interface_with_abstract_hooks() -> None:
    assert MatrixPredictor.input_interface == "matrix"
    assert issubclass(MatrixPredictor, Predictor)
    assert {"_fit_matrix", "_predict_matrix"} <= MatrixPredictor.__abstractmethods__


def test_matrix_predictor_cannot_build_a_design_matrix_without_a_response() -> None:
    class _Recording(MatrixPredictor):
        def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
            raise AssertionError("should not be reached")

        def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
            return np.zeros(len(x))

    batch = ModelInputBatch(
        sample_ids=np.array(["e0", "e1"]),
        drug_ids=_ids(2),
        response=None,
        sample=FeaturizerResult.from_blocks({"view": numeric_feature_block(np.ones((2, 1)))}, np.array(["e0", "e1"])),
        drug=None,
        sample_pair_idx=np.array([0, 1], dtype=np.int64),
        drug_pair_idx=None,
    )

    with pytest.raises(RuntimeError, match="batch.response is required for fit"):
        _Recording()._fit(batch)


def test_registered_neural_network_predictor_uses_only_the_matrix_interface() -> None:
    register_builtin_components()
    cls = get_predictor("neuralNetwork")

    assert issubclass(cls, MatrixPredictor)
    assert not issubclass(cls, BlockPredictor)
    assert cls.input_interface == "matrix"


@pytest.mark.parametrize(
    ("response_length", "pair_idx_length", "message"),
    [(2, 3, "response must be one-dimensional"), (3, 2, "sample_pair_idx must be one-dimensional")],
)
def test_batch_rejects_length_mismatches(response_length: int, pair_idx_length: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ModelInputBatch(
            sample_ids=_ids(3),
            drug_ids=_ids(3),
            response=np.ones(response_length),
            sample=FeaturizerResult(np.empty(0), {}, ()),
            drug=None,
            sample_pair_idx=np.zeros(pair_idx_length, dtype=np.int64),
            drug_pair_idx=None,
        )
