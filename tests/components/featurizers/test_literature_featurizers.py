"""Tests for literature-oriented featurizer registration and contracts."""

from __future__ import annotations

import pytest

from drevalpy.components.contracts import FeatureContract, FeatureFormat, featurizer_contract
from drevalpy.components.register_builtins import register_builtin_components
from drevalpy.components.registry import get_cell_line_featurizer, get_drug_featurizer


@pytest.fixture(autouse=True)
def _register_components() -> None:
    register_builtin_components()


@pytest.mark.parametrize(
    ("name", "expected_format"),
    [
        ("landmarkGenes", FeatureFormat.NUMERIC_MATRIX),
        ("landmarkGenesReduced", FeatureFormat.NUMERIC_MATRIX),
        ("pathways", FeatureFormat.NUMERIC_MATRIX),
        ("bionic", FeatureFormat.NUMERIC_MATRIX),
        ("dipkGeneExpression", FeatureFormat.NUMERIC_MATRIX),
        ("pharmaFormerGeneExpression", FeatureFormat.NUMERIC_MATRIX),
        ("sparsegoOntology", FeatureFormat.NUMERIC_MATRIX),
        ("molirOmics", FeatureFormat.NUMERIC_MATRIX),
        ("superfeltrOmics", FeatureFormat.NUMERIC_MATRIX),
        ("concatFeaturizers", FeatureFormat.NUMERIC_MATRIX),
        ("raw", FeatureFormat.NUMERIC_MATRIX),
        ("pca", FeatureFormat.NUMERIC_MATRIX),
    ],
)
def test_cell_line_literature_featurizer_contracts(
    name: str,
    expected_format: FeatureFormat,
) -> None:
    cls = get_cell_line_featurizer(name)
    contract = featurizer_contract(cls)
    assert isinstance(contract, FeatureContract)
    assert contract.format == expected_format


@pytest.mark.parametrize(
    ("name", "expected_format"),
    [
        ("molgnet", FeatureFormat.RAGGED_SEQUENCE),
        ("bpePharmaformer", FeatureFormat.NUMERIC_MATRIX),
        ("smilesvec", FeatureFormat.NUMERIC_MATRIX),
        ("drugGraph", FeatureFormat.GRAPH),
    ],
)
def test_drug_literature_featurizer_contracts(
    name: str,
    expected_format: FeatureFormat,
) -> None:
    cls = get_drug_featurizer(name)
    contract = featurizer_contract(cls)
    assert isinstance(contract, FeatureContract)
    assert contract.format == expected_format
