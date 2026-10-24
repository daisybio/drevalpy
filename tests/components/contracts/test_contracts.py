"""Tests for internal feature contracts."""

from typing import Any, cast

import pytest

from drevalpy.components.contracts.contracts import (
    FeatureContract,
    FeatureFormat,
    contracts_compatible,
    featurizer_contract,
    normalize_feature_contract,
    predictor_contracts,
)


@pytest.mark.parametrize(
    ("produced", "required", "compatible"),
    [
        (FeatureFormat.NUMERIC_MATRIX, FeatureFormat.NUMERIC_MATRIX, True),
        (FeatureFormat.GRAPH, FeatureFormat.GRAPH, True),
        (FeatureFormat.RAGGED_SEQUENCE, FeatureFormat.RAGGED_SEQUENCE, True),
        (FeatureFormat.GRAPH, FeatureFormat.NUMERIC_MATRIX, False),
    ],
)
def test_contracts_are_compatible_by_format_only(produced, required, compatible: bool) -> None:
    assert contracts_compatible(FeatureContract(format=produced), FeatureContract(format=required)) is compatible


def test_feature_contract_is_frozen_and_normalizes_the_format_shorthand() -> None:
    contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    with pytest.raises(AttributeError):
        cast(Any, contract).format = FeatureFormat.GRAPH
    assert normalize_feature_contract(FeatureFormat.GRAPH) == FeatureContract(format=FeatureFormat.GRAPH)


def test_featurizer_contract_reads_the_canonical_attribute_and_requires_it() -> None:
    class WithContract:
        contract = FeatureContract(format=FeatureFormat.GRAPH)

    class WithoutContract:
        pass

    assert featurizer_contract(WithContract).format == FeatureFormat.GRAPH
    with pytest.raises(TypeError, match="must define a contract"):
        featurizer_contract(WithoutContract)


def test_predictor_contracts_read_the_canonical_attributes_and_require_both() -> None:
    class WithContracts:
        sample_contract = FeatureContract(format=FeatureFormat.RAGGED_SEQUENCE)
        drug_contract = FeatureContract(format=FeatureFormat.GRAPH)

    class MissingDrugContract:
        sample_contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)

    sample, drug = predictor_contracts(WithContracts)
    assert sample.format == FeatureFormat.RAGGED_SEQUENCE
    assert drug.format == FeatureFormat.GRAPH
    with pytest.raises(TypeError, match="must define both"):
        predictor_contracts(MissingDrugContract)
