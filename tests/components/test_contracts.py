"""Tests for internal feature contracts."""

from drevalpy.components.contracts import (
    FeatureContract,
    FeatureKind,
    contracts_compatible,
    featurizer_contract,
    normalize_feature_contract,
    predictor_contracts,
)


def test_dense_contracts_compatible() -> None:
    produced = FeatureContract(kind=FeatureKind.DENSE)
    required = FeatureContract(kind=FeatureKind.DENSE)
    assert contracts_compatible(produced, required)


def test_graph_contracts_compatible_by_kind_only() -> None:
    produced = FeatureContract(kind=FeatureKind.GRAPH)
    required = FeatureContract(kind=FeatureKind.GRAPH)
    assert contracts_compatible(produced, required)


def test_sequence_contracts_compatible() -> None:
    produced = FeatureContract(kind=FeatureKind.SEQUENCE)
    required = FeatureContract(kind=FeatureKind.SEQUENCE)
    assert contracts_compatible(produced, required)


def test_kind_mismatch_is_incompatible() -> None:
    produced = FeatureContract(kind=FeatureKind.GRAPH)
    required = FeatureContract(kind=FeatureKind.DENSE)
    assert not contracts_compatible(produced, required)


def test_feature_contract_is_frozen() -> None:
    contract = FeatureContract(kind=FeatureKind.DENSE)
    try:
        contract.kind = FeatureKind.GRAPH  # type: ignore[misc]
        raised = False
    except AttributeError:
        raised = True
    assert raised


def test_normalize_feature_contract_accepts_kind_shorthand() -> None:
    assert normalize_feature_contract(FeatureKind.GRAPH) == FeatureContract(kind=FeatureKind.GRAPH)


def test_featurizer_contract_prefers_canonical_attribute() -> None:
    class WithContract:
        contract = FeatureContract(kind=FeatureKind.GRAPH)
        output_contract = FeatureContract(kind=FeatureKind.DENSE)

    assert featurizer_contract(WithContract).kind == FeatureKind.GRAPH


def test_predictor_contracts_prefers_canonical_attributes() -> None:
    class WithContracts:
        cell_line_contract = FeatureContract(kind=FeatureKind.SEQUENCE)
        drug_contract = FeatureContract(kind=FeatureKind.GRAPH)
        required_cell_line_contract = FeatureContract(kind=FeatureKind.DENSE)
        required_drug_contract = FeatureContract(kind=FeatureKind.DENSE)

    cell_line, drug = predictor_contracts(WithContracts)
    assert cell_line.kind == FeatureKind.SEQUENCE
    assert drug.kind == FeatureKind.GRAPH
