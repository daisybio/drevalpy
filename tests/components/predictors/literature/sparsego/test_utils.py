"""Tests for SparseGO ontology utilities."""

from __future__ import annotations

import networkx as nx
import pytest

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


def _write_ontology(tmp_path, text: str) -> str:
    path = tmp_path / "ont.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_load_ontology_rejects_term_without_genes(tmp_path) -> None:
    path = _write_ontology(tmp_path, "ROOT TERM default\nTERM GENE1 gene\nROOT LEAF default\n")
    with pytest.raises(ValueError, match="without any annotated gene.*LEAF"):
        load_ontology(path, {"GENE1": 0})


def test_load_ontology_rejects_multiple_roots(tmp_path) -> None:
    path = _write_ontology(tmp_path, "A B default\nB GENE1 gene\nC D default\nD GENE1 gene\n")
    with pytest.raises(ValueError, match="2 roots"):
        load_ontology(path, {"GENE1": 0})


def test_load_ontology_rejects_disconnected_components_with_single_root(tmp_path) -> None:
    path = _write_ontology(
        tmp_path,
        "ROOT X default\nX GENE1 gene\nC1 C2 default\nC2 C1 default\nC1 GENE1 gene\n",
    )
    with pytest.raises(ValueError, match="2 connected components"):
        load_ontology(path, {"GENE1": 0})


def test_load_ontology_rejects_graph_without_term_edges(tmp_path) -> None:
    path = _write_ontology(tmp_path, "TERM GENE1 gene\n")
    with pytest.raises(ValueError, match="no root term"):
        load_ontology(path, {"GENE1": 0})
