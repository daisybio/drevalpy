"""Tests for the maintenance script that builds the autoencoder gene list."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from drevalpy.components.featurizers.sample.gene_lists import _make_autoencoder_gene_list as script
from drevalpy.errors import InvalidInputError

MANIFEST = "[Header]\nx\n[Targets]\nID\tGene\nt1\tAAA\nt2\tBBB\nt3\t\nt4\tCCC\n[Other]\nt5\tZZZ\n"


def _write_dataset(path: Path, values: dict[str, list[float]], unit: str = "log2_tpm_plus_1") -> None:
    genes = list(values)
    matrix = np.array([values[gene] for gene in genes], dtype=np.float32).T
    expression = ad.AnnData(X=matrix, obs=pd.DataFrame(index=[f"s{i}" for i in range(matrix.shape[0])]))
    expression.var_names = genes
    expression.uns["expression_unit"] = unit
    with md.set_options(pull_on_update=False):
        md.MuData({"gene_expression": expression}).write(path)


def test_parse_manifest_reads_the_gene_column_of_the_targets_section() -> None:
    assert script.parse_manifest(MANIFEST) == {"AAA", "BBB", "CCC"}


def test_download_helpers_decode_the_downloaded_files(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("manifest.txt", MANIFEST)
    probemap = b"id\tgene\nE1\tAAA\nE2\tDDD\n"
    monkeypatch.setattr(script, "_download", lambda url, timeout: buffer.getvalue() if "zip" in url else probemap)
    assert script.download_manifest_genes() == {"AAA", "BBB", "CCC"}
    assert script.download_probemap_genes() == {"AAA", "DDD"}


def test_noise_filter_drops_low_mean_and_low_variance_genes_unless_protected() -> None:
    expression = pd.DataFrame(
        {
            "LOW_MEAN": [0.0, 0.1, 0.2, 0.0],
            "FLAT": [5.0, 5.0, 5.0, 5.0],
            "GOOD": [1.0, 4.0, 7.0, 2.0],
            "PROTECTED": [0.0, 0.0, 0.0, 0.0],
        }
    )
    assert script.noise_filter(expression, {"PROTECTED"}) == {"GOOD", "PROTECTED"}


def test_build_chain_narrows_step_by_step() -> None:
    chain = script.build_chain(
        [{"A", "B", "C", "D"}, {"A", "B", "C", "E"}],
        panel={"A", "B", "X"},
        expressed=lambda genes: genes - {"B"},
        tcga_symbols={"A", "B"},
    )
    assert chain["expression_intersection"] == {"A", "B", "C"}
    assert chain["panel_intersection"] == {"A", "B"}
    assert chain["expressed_genes"] == {"A"}
    assert chain[script.GENE_LIST_NAME] == {"A"}


def test_build_chain_needs_a_dataset() -> None:
    with pytest.raises(InvalidInputError, match="At least one"):
        script.build_chain([], set(), lambda genes: genes, set())


def test_create_gene_list_writes_the_surviving_symbols(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    good = [1.0, 4.0, 7.0, 2.0]
    _write_dataset(tmp_path / "ONE.h5mu", {"AAA": good, "BBB": good, "FLAT": [5.0] * 4, "ONLY_ONE": good})
    _write_dataset(tmp_path / "TWO.h5mu", {"AAA": good, "BBB": good, "FLAT": [5.0] * 4}, unit="log2_cpm")
    monkeypatch.setattr(script, "download_manifest_genes", lambda: {"AAA", "FLAT"})
    monkeypatch.setattr(script, "download_probemap_genes", lambda: {"AAA", "FLAT"})
    out = tmp_path / "out"
    out.mkdir()

    genes = script.create_gene_list(UPath(tmp_path), ("ONE", "TWO"), output=UPath(out))

    assert genes == {"AAA"}
    written = pd.read_parquet(out / f"{script.GENE_LIST_NAME}.parquet")
    assert written["Symbol"].tolist() == ["AAA"]


def test_create_gene_list_rejects_a_noise_dataset_that_is_not_log_tpm(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "ONE.h5mu", {"AAA": [1.0, 2.0]}, unit="log2_cpm")
    with pytest.raises(InvalidInputError, match="expression_unit 'log2_tpm_plus_1'"):
        script.create_gene_list(UPath(tmp_path), ("ONE",))


def test_main_passes_the_command_line_to_create_gene_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(script, "create_gene_list", lambda *args, **kwargs: calls.append((args, kwargs)))
    script.main(["--data-path", str(tmp_path), "--datasets", "A", "B", "--noise-dataset", "B", "--output-dir", "o"])
    ((args, kwargs),) = calls
    assert args[1] == ("A", "B")
    assert kwargs["noise_dataset"] == "B"
    assert str(kwargs["output"]) == "o"
