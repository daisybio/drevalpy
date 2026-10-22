"""Tests for the script that builds the gene lists of the TCGA autoencoder gene space."""

import runpy
import sys

import numpy as np
import pandas as pd
import pytest

from drevalpy.datasets.featurizer import create_autoencoder_gene_list as script


def test_noise_filter_drops_unexpressed_and_constant_genes_unless_protected():
    """Low mean or low variance of log2(x + 1) drops a gene; protected genes stay."""
    rng = np.random.default_rng(0)
    expression = pd.DataFrame(
        {
            "expressed": rng.uniform(4, 8, 50),
            "unexpressed": np.zeros(50),
            "constant": np.full(50, 6.0),
            "driver_unexpressed": np.zeros(50),
        }
    )
    assert script.noise_filter(expression, set()) == {"expressed"}
    assert script.noise_filter(expression, {"driver_unexpressed", "not_in_data"}) == {"expressed", "driver_unexpressed"}


def test_build_chain_is_a_chain_of_subsets():
    """Each step keeps only genes that survive all steps before it."""
    chain = script.build_chain(
        expression=[{"A", "B", "C", "D", "E"}, {"A", "B", "C", "D", "F"}],
        panel={"A", "B", "C", "X"},
        expressed=lambda genes: genes - {"C"},
        tcga_symbols={"A", "Z"},
    )
    assert chain["ctrpv2_gdsc1_intersection"] == {"A", "B", "C", "D"}
    assert chain["ctrpv2_gdsc1_iontorrent_intersection"] == {"A", "B", "C"}
    assert chain["autoencoder_expressed_genes"] == {"A", "B"}
    assert chain["tcga_autoencoder_genes_available"] == {"A"}
    with pytest.raises(ValueError):
        script.build_chain([], set(), lambda genes: genes, set())


def test_readers(tmp_path):
    """
    Expression columns without the id columns, panel symbols from the right column, Symbol lists.

    :param tmp_path: pytest temporary directory
    """
    pd.DataFrame({"cell_line_name": ["a"], "cellosaurus_id": ["x"], "G1": [1.0], "G2": [2.0]}).to_csv(
        tmp_path / "expr.csv", index=False
    )
    assert script.expression_genes(tmp_path / "expr.csv") == {"G1", "G2"}
    (tmp_path / "panel.bed").write_text('"chr1";"1";"2";"n";"info";"35";"G1"\n"ERCC-1";"1";"2";"n";"i";"3";"ERCC-1"\n')
    assert script.read_panel_genes(tmp_path / "panel.bed") == {"G1", "ERCC-1"}
    pd.DataFrame({"Symbol": ["G1", "G3"]}).to_csv(tmp_path / "list.csv", index=False)
    assert script.read_symbols(tmp_path / "list.csv") == {"G1", "G3"}


def test_main_writes_the_four_lists(tmp_path, monkeypatch, capsys):
    """
    End to end on small files: four sorted lists with a Symbol column, the dropped gene is reported.

    :param capsys: pytest capture fixture
    :param monkeypatch: pytest monkeypatch fixture
    :param tmp_path: pytest temporary directory
    """
    rng = np.random.default_rng(1)
    genes = {"D1": ["A", "B", "C", "D"], "D2": ["A", "B", "C", "E"]}
    for name, columns in genes.items():
        (tmp_path / name).mkdir()
        frame = pd.DataFrame({g: rng.uniform(4, 8, 30) for g in columns})
        frame["C"] = 0.0  # unexpressed
        frame.insert(0, "cell_line_name", [f"c{i}" for i in range(30)])
        frame.to_csv(tmp_path / name / "gene_expression.csv", index=False)
    (tmp_path / "panel.bed").write_text("".join(f'"c";"1";"2";"n";"i";"3";"{g}"\n' for g in ["A", "B", "C"]))
    pd.DataFrame({"Symbol": ["A"]}).to_csv(tmp_path / "protect.csv", index=False)
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
            "--noise_dataset",
            "D1",
            "--panel",
            str(tmp_path / "panel.bed"),
        ]
        + ["--protect", str(tmp_path / "protect.csv"), "--probemap", str(tmp_path / "probemap.tsv")],
    )
    script.main()
    directory = tmp_path / "meta" / "gene_lists"
    expected = {
        "ctrpv2_gdsc1_intersection": ["A", "B", "C"],
        "ctrpv2_gdsc1_iontorrent_intersection": ["A", "B", "C"],
        "autoencoder_expressed_genes": ["A", "B"],
        "tcga_autoencoder_genes_available": ["B"],
    }
    for name, expected_genes in expected.items():
        assert pd.read_csv(directory / f"{name}.csv")["Symbol"].tolist() == expected_genes
    assert "dropped 1: ['A']" in capsys.readouterr().out


MANIFEST = "[Header]\nx\ty\n\n[Targets]\nID\tGene\tother\na1\tG1\t1\na2\tG2\t2\na3\t\t3\n\n[Other]\nID\tGene\nb\tG9\n"


def test_parse_manifest_reads_only_the_targets_section():
    """Gene symbols come from the Gene column of [Targets]; empty cells and later sections are ignored."""
    assert script.parse_manifest(MANIFEST) == {"G1", "G2"}


def test_download_manifest_genes(tmp_path):
    """
    The zip is read from a url (here a local file) and parsed.

    :param tmp_path: pytest temporary directory
    """
    import zipfile

    archive = tmp_path / "manifest.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("manifest.txt", MANIFEST)
    assert script.download_manifest_genes(archive.as_uri()) == {"G1", "G2"}


def test_download_probemap_genes(tmp_path):
    """
    The probemap is read from a url (here a local file) and its gene column returned.

    :param tmp_path: pytest temporary directory
    """
    probemap = tmp_path / "probemap.tsv"
    pd.DataFrame({"id": ["l1", "l2"], "gene": ["B", "Q"]}).to_csv(probemap, sep="\t", index=False)
    assert script.download_probemap_genes(probemap.as_uri()) == {"B", "Q"}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_script_starts_as_a_module(monkeypatch):
    """The script can be started with ``python -m`` and prints its usage.

    :param monkeypatch: pytest monkeypatch fixture
    """
    monkeypatch.setattr(sys, "argv", [script.__name__, "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module(script.__name__, run_name="__main__")
    assert exit_info.value.code == 0
