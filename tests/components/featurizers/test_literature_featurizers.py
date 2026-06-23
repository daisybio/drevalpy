"""Tests for literature-oriented featurizer registration and contracts."""

from __future__ import annotations

import pytest

from drevalpy.components.contracts import FeatureContract, FeatureKind
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components(include_legacy=False)


@pytest.mark.parametrize(
    ("name", "expected_kind", "expected_view"),
    [
        ("landmarkGenes", FeatureKind.DENSE, "gene_expression"),
        ("landmarkGenesReduced", FeatureKind.DENSE, "gene_expression"),
        ("pathways", FeatureKind.DENSE, "pathways"),
        ("bionic", FeatureKind.DENSE, "bionic_features"),
        ("multiViewStructured", FeatureKind.DENSE, None),
    ],
)
def test_cell_line_literature_featurizer_contracts(
    name: str,
    expected_kind: FeatureKind,
    expected_view: str | None,
) -> None:
    cls = get_cell_line_featurizer(name)
    contract = cls.output_contract
    assert isinstance(contract, FeatureContract)
    assert contract.kind == expected_kind
    if expected_view is not None:
        assert contract.view == expected_view
    if name == "multiViewStructured":
        assert contract.scope == "multi_view"


@pytest.mark.parametrize(
    ("name", "expected_kind", "expected_view"),
    [
        ("molgnet", FeatureKind.DENSE, "molgnet_features"),
        ("bpePharmaformer", FeatureKind.DENSE, "bpe_smiles"),
        ("smilesvec", FeatureKind.DENSE, "smilesvec"),
        ("drugGraph", FeatureKind.GRAPH, "drug_graph"),
    ],
)
def test_drug_literature_featurizer_contracts(
    name: str,
    expected_kind: FeatureKind,
    expected_view: str | None,
) -> None:
    cls = get_drug_featurizer(name)
    contract = cls.output_contract
    assert isinstance(contract, FeatureContract)
    assert contract.kind == expected_kind
    if expected_view is not None:
        assert contract.view == expected_view
