"""Tests for the script that builds the gene lists of the TCGA autoencoder gene space."""

import sys

import pandas as pd
import pytest

from drevalpy.datasets.featurizer import create_autoencoder_gene_list as script


def test_build_chain_is_a_chain_of_subsets():
    """Each step keeps only genes that survive all steps before it."""
    chain = script.build_chain(
        expression=[{"A", "B", "C", "D", "E"}, {"A", "B", "C", "D", "F"}],
        panel={"A", "B", "C", "X"},
        vocabulary={"A", "B", "Y"},
        tcga_symbols={"A", "Z"},
    )
    assert chain["ctrpv2_gdsc1_intersection"] == {"A", "B", "C", "D"}
    assert chain["ctrpv2_gdsc1_iontorrent_intersection"] == {"A", "B", "C"}
    assert chain["autoencoder_expressed_genes"] == {"A", "B"}
    assert chain["tcga_autoencoder_genes_available"] == {"A"}
    with pytest.raises(ValueError):
        script.build_chain([], set(), set(), set())


def test_readers(tmp_path):
    """Expression columns without the id columns, panel symbols from the right column, Symbol lists."""
    pd.DataFrame({"cell_line_name": ["a"], "cellosaurus_id": ["x"], "G1": [1.0], "G2": [2.0]}).to_csv(
        tmp_path / "expr.csv", index=False
    )
    assert script.expression_genes(tmp_path / "expr.csv") == {"G1", "G2"}
    (tmp_path / "panel.bed").write_text('"chr1";"1";"2";"n";"info";"35";"G1"\n"ERCC-1";"1";"2";"n";"i";"3";"ERCC-1"\n')
    assert script.read_panel_genes(tmp_path / "panel.bed") == {"G1", "ERCC-1"}
    pd.DataFrame({"Symbol": ["G1", "G3"]}).to_csv(tmp_path / "list.csv", index=False)
    assert script.read_symbols(tmp_path / "list.csv") == {"G1", "G3"}


def test_main_writes_the_four_lists(tmp_path, monkeypatch, capsys):
    """End to end on small files: four sorted lists with a Symbol column, the dropped gene is reported."""
    for name, genes in {"D1": ["A", "B", "C", "D"], "D2": ["A", "B", "C", "E"]}.items():
        (tmp_path / name).mkdir()
        pd.DataFrame({"cell_line_name": ["a"], **{g: [1.0] for g in genes}}).to_csv(
            tmp_path / name / "gene_expression.csv", index=False
        )
    (tmp_path / "panel.bed").write_text("".join(f'"c";"1";"2";"n";"i";"3";"{g}"\n' for g in ["A", "B", "C"]))
    pd.DataFrame({"Symbol": ["C", "B"]}).to_csv(tmp_path / "vocabulary.csv", index=False)
    pd.DataFrame({"id": ["l1", "l2"], "gene": ["B", "Q"]}).to_csv(tmp_path / "probemap.tsv", sep="\t", index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--data_path",
            str(tmp_path),
            "--datasets",
            "D1",
            "D2",
            "--panel",
            str(tmp_path / "panel.bed"),
            "--vocabulary",
            str(tmp_path / "vocabulary.csv"),
            "--probemap",
            str(tmp_path / "probemap.tsv"),
        ],
    )
    script.main()
    directory = tmp_path / "meta" / "gene_lists"
    expected = {
        "ctrpv2_gdsc1_intersection": ["A", "B", "C"],
        "ctrpv2_gdsc1_iontorrent_intersection": ["A", "B", "C"],
        "autoencoder_expressed_genes": ["B", "C"],
        "tcga_autoencoder_genes_available": ["B"],
    }
    for name, genes in expected.items():
        assert pd.read_csv(directory / f"{name}.csv")["Symbol"].tolist() == genes
    assert "dropped 1: ['C']" in capsys.readouterr().out
