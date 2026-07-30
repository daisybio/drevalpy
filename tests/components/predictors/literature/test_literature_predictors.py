"""Tests for literature predictor registration and contracts."""

from __future__ import annotations

import pytest

from drevalpy.components.contracts import FeatureFormat
from drevalpy.components.predictors.matrix import MatrixPredictor
from drevalpy.components.predictors.structured import BlockPredictor
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.components.registry import get_predictor


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


@pytest.mark.parametrize(
    ("name", "interface", "requires_drug"),
    [
        ("drugGNN", "block", True),
        ("neuralNetwork", "matrix", True),
        ("precily", "block", True),
        ("srmf", "block", True),
        ("molir", "block", True),
        ("superfeltr", "block", True),
        ("pharmaFormer", "block", True),
        ("dipk", "block", True),
        ("sparsego", "block", True),
    ],
)
def test_literature_predictor_flags(name: str, interface: str, requires_drug: bool) -> None:
    cls = get_predictor(name)
    if interface == "matrix":
        assert issubclass(cls, MatrixPredictor)
        assert not issubclass(cls, BlockPredictor)
    elif interface == "block":
        assert issubclass(cls, BlockPredictor)
        assert not issubclass(cls, MatrixPredictor)
    assert getattr(cls, "requires_drug_featurizer", True) is requires_drug


def test_druggnn_requires_graph_drug_contract() -> None:
    cls = get_predictor("drugGNN")
    assert cls.drug_contract.format == FeatureFormat.GRAPH
    assert cls.required_drug_blocks == ("drug_graph",)
    assert cls.supports_early_stopping is True


def test_neural_network_requires_numeric_contracts() -> None:
    cls = get_predictor("neuralNetwork")
    assert cls.cell_line_contract.format == FeatureFormat.NUMERIC_MATRIX
    assert cls.drug_contract.format == FeatureFormat.NUMERIC_MATRIX
