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


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("ROOT TERM default\nTERM GENE1 gene\nROOT LEAF default\n", "without any annotated gene.*LEAF"),
        ("A B default\nB GENE1 gene\nC D default\nD GENE1 gene\n", "2 roots"),
        (
            "ROOT X default\nX GENE1 gene\nC1 C2 default\nC2 C1 default\nC1 GENE1 gene\n",
            "2 connected components",
        ),
        ("TERM GENE1 gene\n", "no root term"),
    ],
    ids=["term-without-genes", "multiple-roots", "disconnected", "no-root"],
)
def test_load_ontology_rejects_malformed_ontologies(tmp_path, text: str, message: str) -> None:
    path = tmp_path / "ont.txt"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_ontology(str(path), {"GENE1": 0})
