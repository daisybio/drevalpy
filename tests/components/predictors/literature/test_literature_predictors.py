"""Tests for literature predictor registration and contracts."""

from __future__ import annotations

import pytest

from drevalpy.components.contracts import FeatureKind
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.components.registry import get_predictor


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components(include_legacy=False)


@pytest.mark.parametrize(
    ("name", "structured", "requires_drug"),
    [
        ("drugGNN", True, True),
        ("neuralNetwork", False, True),
        ("precily", True, True),
        ("srmf", True, True),
        ("molir", True, False),
        ("superfeltr", True, False),
        ("pharmaFormer", True, True),
        ("dipk", True, True),
    ],
)
def test_literature_predictor_flags(name: str, structured: bool, requires_drug: bool) -> None:
    cls = get_predictor(name)
    assert getattr(cls, "uses_structured_features", False) is structured
    assert getattr(cls, "requires_drug_featurizer", True) is requires_drug


def test_druggnn_requires_graph_drug_contract() -> None:
    cls = get_predictor("drugGNN")
    assert cls.required_drug_contract.kind == FeatureKind.GRAPH


def test_neural_network_requires_dense_contracts() -> None:
    cls = get_predictor("neuralNetwork")
    assert cls.required_cell_line_contract.kind == FeatureKind.DENSE
    assert cls.required_drug_contract.kind == FeatureKind.DENSE
