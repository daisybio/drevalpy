"""Tests for internal feature contracts."""

from drevalpy.components.contracts import FeatureContract, FeatureKind, contracts_compatible


def test_dense_contracts_compatible_by_kind_only() -> None:
    produced = FeatureContract(kind=FeatureKind.DENSE, view="gene_expression")
    required = FeatureContract(kind=FeatureKind.DENSE, view="proteomics")
    assert contracts_compatible(produced, required)


def test_graph_contracts_require_backend_match_when_required() -> None:
    produced = FeatureContract(
        kind=FeatureKind.GRAPH,
        backend="pyg",
        scope="per_drug",
        has_node_features=True,
        has_edge_features=True,
    )
    required = FeatureContract(
        kind=FeatureKind.GRAPH,
        backend="pyg",
        scope="per_drug",
        has_node_features=True,
        has_edge_features=True,
    )
    assert contracts_compatible(produced, required)


def test_graph_contracts_reject_backend_mismatch() -> None:
    produced = FeatureContract(kind=FeatureKind.GRAPH, backend="pyg")
    required = FeatureContract(kind=FeatureKind.GRAPH, backend="dgl")
    assert not contracts_compatible(produced, required)


def test_graph_contracts_reject_scope_mismatch() -> None:
    produced = FeatureContract(kind=FeatureKind.GRAPH, scope="per_drug")
    required = FeatureContract(kind=FeatureKind.GRAPH, scope="global")
    assert not contracts_compatible(produced, required)


def test_graph_contracts_reject_node_feature_mismatch() -> None:
    produced = FeatureContract(kind=FeatureKind.GRAPH, has_node_features=True)
    required = FeatureContract(kind=FeatureKind.GRAPH, has_node_features=False)
    assert not contracts_compatible(produced, required)


def test_kind_mismatch_is_incompatible() -> None:
    produced = FeatureContract(kind=FeatureKind.GRAPH)
    required = FeatureContract(kind=FeatureKind.DENSE)
    assert not contracts_compatible(produced, required)


def test_feature_contract_is_frozen() -> None:
    contract = FeatureContract(kind=FeatureKind.DENSE)
    try:
        contract.view = "gene_expression"  # type: ignore[misc]
        raised = False
    except AttributeError:
        raised = True
    assert raised
