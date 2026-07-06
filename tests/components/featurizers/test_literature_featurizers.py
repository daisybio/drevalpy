"""Tests for literature-oriented featurizer registration and contracts."""

from __future__ import annotations

import pytest

from drevalpy.components.contracts import FeatureContract, FeatureKind, featurizer_contract
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components(include_legacy=False)


@pytest.mark.parametrize(
    ("name", "expected_kind"),
    [
        ("landmarkGenes", FeatureKind.DENSE),
        ("landmarkGenesReduced", FeatureKind.DENSE),
        ("pathways", FeatureKind.DENSE),
        ("bionic", FeatureKind.DENSE),
        ("concatFeaturizers", FeatureKind.DENSE),
        ("geneExpression", FeatureKind.DENSE),
        ("mutations", FeatureKind.DENSE),
        ("methylationPCA", FeatureKind.DENSE),
    ],
)
def test_cell_line_literature_featurizer_contracts(
    name: str,
    expected_kind: FeatureKind,
) -> None:
    cls = get_cell_line_featurizer(name)
    contract = featurizer_contract(cls)
    assert isinstance(contract, FeatureContract)
    assert contract.kind == expected_kind


@pytest.mark.parametrize(
    ("name", "expected_kind"),
    [
        ("molgnet", FeatureKind.DENSE),
        ("bpePharmaformer", FeatureKind.DENSE),
        ("smilesvec", FeatureKind.DENSE),
        ("drugGraph", FeatureKind.GRAPH),
    ],
)
def test_drug_literature_featurizer_contracts(
    name: str,
    expected_kind: FeatureKind,
) -> None:
    cls = get_drug_featurizer(name)
    contract = featurizer_contract(cls)
    assert isinstance(contract, FeatureContract)
    assert contract.kind == expected_kind
