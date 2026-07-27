"""In-memory tests for sparsego_graph pruning helpers."""

from __future__ import annotations

import networkx as nx

from drevalpy.datasets.featurizer.sparsego_graph import (
    GO_ROOT,
    _direct_genes_and_children,
    _should_remove_nm_node,
    build_level_list,
)


def test_build_level_list_peels_leaves() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([(GO_ROOT, "GO:1"), ("GO:1", "GENE1")])
    levels = build_level_list(graph)
    assert "GENE1" in levels[0]


def test_should_remove_nm_node_when_too_few_genes() -> None:
    graph = nx.DiGraph()
    graph.add_edges_from([("GO:1", "GENE1")])
    genes, children = _direct_genes_and_children(graph, "GO:1")
    assert _should_remove_nm_node(graph, "GO:1", genes, children, n=5, m=10)
