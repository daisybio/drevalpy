"""Tests for SparseGO ontology utilities."""

from __future__ import annotations

import networkx as nx

from drevalpy.components.predictors.literature.sparsego.utils import load_ontology


def test_load_ontology_builds_graph_and_pair_arrays(tmp_path) -> None:
    ont_path = tmp_path / "sparseGO_ont.txt"
    ont_path.write_text(
        "ROOT TERM default\nTERM GENE1 gene\n",
        encoding="utf-8",
    )
    gene2id = {"GENE1": 0}

    ontology_graph, terms_pairs, genes_terms_pairs = load_ontology(str(ont_path), gene2id)

    assert isinstance(ontology_graph, nx.DiGraph)
    assert ontology_graph.has_edge("ROOT", "TERM")
    assert terms_pairs.shape == (1, 2)
    assert genes_terms_pairs.shape == (1, 2)
    assert genes_terms_pairs[0].tolist() == ["TERM", "GENE1"]
