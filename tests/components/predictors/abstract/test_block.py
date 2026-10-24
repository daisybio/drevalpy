"""Tests for the block-interface predictor base."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from drevalpy.components.predictors.abstract.base import Predictor
from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.abstract.matrix import MatrixPredictor
from drevalpy.registry._builtins import register_builtin_components
from drevalpy.registry.predictor import get as get_predictor
from drevalpy.types.data.batch.model_input_batch import ModelInputBatch

BLOCK_PREDICTOR_NAMES = (
    "drugGNN",
    "precily",
    "srmf",
    "molir",
    "superfeltr",
    "pharmaFormer",
    "dipk",
    "sparsego",
)


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


def test_block_predictor_is_an_abstract_block_interface_distinct_from_matrix() -> None:
    assert BlockPredictor.input_interface == "block"
    assert issubclass(BlockPredictor, Predictor)
    assert not issubclass(BlockPredictor, MatrixPredictor)
    assert {"_fit", "_predict"} <= BlockPredictor.__abstractmethods__
    with pytest.raises(TypeError):
        BlockPredictor()


def test_block_predictor_subclass_supplying_both_hooks_is_concrete() -> None:
    class _Constant(BlockPredictor):
        def _fit(self, batch: ModelInputBatch) -> None:
            self._value = 1.0

        def _predict(self, batch: ModelInputBatch) -> np.ndarray:
            return np.full(len(batch.sample_ids), self._value)

    assert not inspect.isabstract(_Constant)
    assert _Constant.input_interface == "block"


@pytest.mark.parametrize("name", BLOCK_PREDICTOR_NAMES)
def test_registered_block_predictors_use_only_the_block_interface(name: str) -> None:
    cls = get_predictor(name)

    assert issubclass(cls, BlockPredictor)
    assert not issubclass(cls, MatrixPredictor)
    assert cls.input_interface == "block"
