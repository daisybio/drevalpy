"""Tests for the matrix predictor interface and the ``ModelInputBatch`` contract it relies on."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch


def _ids(n: int) -> np.ndarray:
    return np.array([f"id_{i}" for i in range(n)])


def test_matrix_predictor_declares_the_matrix_input_interface() -> None:
    assert MatrixPredictor.input_interface == "matrix"
    assert issubclass(MatrixPredictor, Predictor)


@pytest.mark.parametrize("method_name", ["_fit_matrix", "_predict_matrix"])
def test_matrix_predictor_leaves_the_matrix_hooks_abstract(method_name: str) -> None:
    assert method_name in MatrixPredictor.__abstractmethods__


def test_matrix_predictor_cannot_build_a_design_matrix_without_a_response() -> None:
    class _Recording(MatrixPredictor):
        def _fit_matrix(self, x: np.ndarray, y: np.ndarray) -> None:
            raise AssertionError("should not be reached")

        def _predict_matrix(self, x: np.ndarray) -> np.ndarray:
            return np.zeros(len(x))

    batch = ModelInputBatch(
        cell_line_ids=_ids(2),
        drug_ids=_ids(2),
        response=None,
        cell_line_entity_ids=np.array(["e0", "e1"]),
        drug_entity_ids=None,
        cell_line_features=np.ones((2, 1)),
        drug_features=None,
        cell_line_pair_idx=np.array([0, 1], dtype=np.int64),
        drug_pair_idx=None,
    )

    with pytest.raises(ValueError, match="response is required to build a feature matrix"):
        _Recording()._fit(batch)


def test_registered_neural_network_predictor_uses_only_the_matrix_interface() -> None:
    register_builtin_components()

    cls = get_predictor("neuralNetwork")

    assert issubclass(cls, MatrixPredictor)
    assert not issubclass(cls, BlockPredictor)
    assert cls.input_interface == "matrix"


def test_batch_rejects_response_length_mismatch() -> None:
    with pytest.raises(ValueError, match="response length"):
        ModelInputBatch(
            cell_line_ids=_ids(3),
            drug_ids=_ids(3),
            response=np.ones(2),
            cell_line_entity_ids=np.empty(0),
            drug_entity_ids=None,
            cell_line_features=np.empty((0, 0)),
            drug_features=None,
            cell_line_pair_idx=np.zeros(3, dtype=np.int64),
            drug_pair_idx=None,
        )


def test_batch_rejects_pair_idx_length_mismatch() -> None:
    with pytest.raises(ValueError, match="cell_line_pair_idx length"):
        ModelInputBatch(
            cell_line_ids=_ids(3),
            drug_ids=_ids(3),
            response=np.ones(3),
            cell_line_entity_ids=np.empty(0),
            drug_entity_ids=None,
            cell_line_features=np.empty((0, 0)),
            drug_features=None,
            cell_line_pair_idx=np.zeros(2, dtype=np.int64),
            drug_pair_idx=None,
        )
