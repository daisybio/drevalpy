"""Tests for internal feature contracts."""

from drevalpy.components.contracts import (
    FeatureContract,
    FeatureFormat,
    contracts_compatible,
    featurizer_contract,
    normalize_feature_contract,
    predictor_contracts,
)


def test_numeric_contracts_compatible() -> None:
    produced = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    required = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    assert contracts_compatible(produced, required)


def test_graph_contracts_compatible_by_format_only() -> None:
    produced = FeatureContract(format=FeatureFormat.GRAPH)
    required = FeatureContract(format=FeatureFormat.GRAPH)
    assert contracts_compatible(produced, required)


def test_ragged_contracts_compatible() -> None:
    produced = FeatureContract(format=FeatureFormat.RAGGED_SEQUENCE)
    required = FeatureContract(format=FeatureFormat.RAGGED_SEQUENCE)
    assert contracts_compatible(produced, required)


def test_format_mismatch_is_incompatible() -> None:
    produced = FeatureContract(format=FeatureFormat.GRAPH)
    required = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    assert not contracts_compatible(produced, required)


def test_feature_contract_is_frozen() -> None:
    contract = FeatureContract(format=FeatureFormat.NUMERIC_MATRIX)
    try:
        contract.format = FeatureFormat.GRAPH  # type: ignore[misc]
        raised = False
    except AttributeError:
        raised = True
    assert raised


def test_normalize_feature_contract_accepts_format_shorthand() -> None:
    assert normalize_feature_contract(FeatureFormat.GRAPH) == FeatureContract(format=FeatureFormat.GRAPH)


def test_featurizer_contract_reads_canonical_attribute() -> None:
    class WithContract:
        contract = FeatureContract(format=FeatureFormat.GRAPH)

    assert featurizer_contract(WithContract).format == FeatureFormat.GRAPH


def test_predictor_contracts_reads_canonical_attributes() -> None:
    class WithContracts:
        cell_line_contract = FeatureContract(format=FeatureFormat.RAGGED_SEQUENCE)
        drug_contract = FeatureContract(format=FeatureFormat.GRAPH)

    cell_line, drug = predictor_contracts(WithContracts)
    assert cell_line.format == FeatureFormat.RAGGED_SEQUENCE
    assert drug.format == FeatureFormat.GRAPH
