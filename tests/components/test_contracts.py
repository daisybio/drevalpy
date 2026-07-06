"""Tests for internal feature contracts."""

from drevalpy.components.contracts import FeatureContract, FeatureKind, contracts_compatible


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
