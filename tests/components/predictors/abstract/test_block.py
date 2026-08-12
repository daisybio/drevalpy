"""Tests for the block-interface predictor base.

The per-predictor interface assertions were carved out of
``literature/test_init.py``, which now keeps only the package-boundary policy
checks; the interface itself is a property of ``abstract/block.py``.
"""

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


def test_block_predictor_declares_the_block_input_interface() -> None:
    assert BlockPredictor.input_interface == "block"


def test_block_predictor_derives_from_the_shared_predictor_base() -> None:
    assert issubclass(BlockPredictor, Predictor)


def test_block_predictor_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        BlockPredictor()


@pytest.mark.parametrize("method_name", ["_fit", "_predict"])
def test_block_predictor_leaves_fit_and_predict_abstract(method_name: str) -> None:
    assert method_name in BlockPredictor.__abstractmethods__
    assert getattr(BlockPredictor, method_name).__isabstractmethod__ is True


def test_block_predictor_does_not_flatten_batches_like_the_matrix_base() -> None:
    assert not issubclass(BlockPredictor, MatrixPredictor)
    assert BlockPredictor.input_interface != MatrixPredictor.input_interface


def test_block_predictor_subclass_supplying_both_hooks_is_concrete() -> None:
    class _Constant(BlockPredictor):
        def _fit(self, batch: ModelInputBatch) -> None:
            self._value = 1.0

        def _predict(self, batch: ModelInputBatch) -> np.ndarray:
            return np.full(len(batch.cell_line_ids), self._value)

    assert not inspect.isabstract(_Constant)
    assert _Constant.input_interface == "block"


@pytest.mark.parametrize("name", BLOCK_PREDICTOR_NAMES)
def test_registered_block_predictors_use_only_the_block_interface(name: str) -> None:
    cls = get_predictor(name)

    assert issubclass(cls, BlockPredictor)
    assert not issubclass(cls, MatrixPredictor)
    assert cls.input_interface == "block"
