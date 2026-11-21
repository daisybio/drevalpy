"""Tests for the sample gene-list helpers and the autoencoder gene-list builder."""

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

from drevalpy.components.featurizers.sample.gene_lists import (
    GENE_LISTS_BY_VIEW,
    GENE_LISTS_DIR,
    checked_view_gene_lists,
    gene_names_from_list,
    resolve_gene_list,
    resolve_gene_list_path,
    select_genes,
    view_gene_union,
)
from drevalpy.components.featurizers.sample.gene_lists import _make_autoencoder_gene_list as script
from drevalpy.errors import InvalidInputError, UnknownNameError
from tests.conftest import MockFeatureSource


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
@pytest.mark.parametrize(("column", "expected"), [("Symbol", ["A", "B"]), ("gene_name", ["A", "B"]), ("other", None)])
def test_gene_names_read_known_columns_and_reject_unknown(tmp_path: UPath, suffix: str, column: str, expected) -> None:
    path = tmp_path / f"genes{suffix}"
    frame = pd.DataFrame({column: ["A", "B"]})
    frame.to_parquet(path, index=False) if suffix == ".parquet" else frame.to_csv(path, index=False)
    if expected is None:
        with pytest.raises(InvalidInputError, match="recognized gene-name column"):
            gene_names_from_list(path)
    else:
        assert gene_names_from_list(path) == expected


def test_resolve_gene_list_path_uses_the_packaged_file() -> None:
    path = resolve_gene_list_path("landmark_genes")
    assert path.is_file()
    assert path == GENE_LISTS_DIR / "landmark_genes.parquet"


def test_resolve_gene_list_path_ignores_cache_dir(tmp_path: UPath, monkeypatch: pytest.MonkeyPatch) -> None:
    gene_dir = tmp_path / "meta" / "gene_lists"
    gene_dir.mkdir(parents=True)
    pd.DataFrame({"Symbol": ["G1"]}).to_parquet(gene_dir / "landmark_genes.parquet", index=False)
    monkeypatch.setenv("DREVALPY_CACHE_DIR", str(tmp_path))
    assert resolve_gene_list_path("landmark_genes") == GENE_LISTS_DIR / "landmark_genes.parquet"


def test_resolve_gene_list_path_lists_available_lists_when_missing() -> None:
    with pytest.raises(UnknownNameError, match="Available gene lists: .*landmark_genes"):
        resolve_gene_list_path("not_a_gene_list")


def _source(view: str, names: tuple[str, ...] | None) -> MockFeatureSource:
    return MockFeatureSource(features={"s": {view: [0.0] * len(names or ())}}, meta_info={view: names} if names else {})


def test_resolve_gene_list_returns_the_symbols_in_file_order() -> None:
    genes = resolve_gene_list("landmark_genes")
    assert genes[:2] == ("INSIG1", "FOXO3")
    assert len(genes) == len(set(genes)) == 867


def test_resolve_gene_list_rejects_an_unknown_name() -> None:
    with pytest.raises(UnknownNameError, match="not_a_list.*Available gene lists"):
        resolve_gene_list("not_a_list")


def test_select_genes_follows_the_list_order_and_ignores_extra_genes() -> None:
    genes = resolve_gene_list("landmark_genes_reduced")
    source = _source("gene_expression", ("EXTRA", *reversed(genes)))
    assert select_genes(source, "gene_expression", "landmark_genes_reduced") == genes


def test_select_genes_names_the_list_the_view_and_the_missing_count() -> None:
    genes = resolve_gene_list("landmark_genes")
    source = _source("gene_expression", genes[:55])
    with pytest.raises(InvalidInputError) as raised:
        select_genes(source, "gene_expression", "landmark_genes")
    message = str(raised.value)
    assert "view 'gene_expression' lacks 812 of 867 genes in 'landmark_genes'" in message
    assert genes[55] in message
    assert "gene_list=None" in message


def test_select_genes_requires_feature_names() -> None:
    with pytest.raises(InvalidInputError, match="no feature names for view 'gene_expression'"):
        select_genes(_source("gene_expression", None), "gene_expression", "landmark_genes")


def test_every_shipped_list_belongs_to_a_view() -> None:
    shipped = {path.stem for path in GENE_LISTS_DIR.glob("*.parquet")}
    assigned = {name for names in GENE_LISTS_BY_VIEW.values() for name in names}
    assert shipped == assigned


def test_autoencoder_list_is_shipped_and_inside_the_expression_view() -> None:
    genes = resolve_gene_list("tcga_autoencoder_genes_available")
    assert len(genes) > 5000
    assert set(genes) <= set(view_gene_union("gene_expression"))


def test_view_gene_union_keeps_first_appearance_order_without_repeats() -> None:
    union = view_gene_union("gene_expression")
    assert len(union) == len(set(union))
    assert union[: len(resolve_gene_list("landmark_genes"))] == resolve_gene_list("landmark_genes")
    assert view_gene_union("unknown_view") == ()


def test_checked_view_gene_lists_copies_and_rejects_unknown_views() -> None:
    given = {"gene_expression": "landmark_genes"}
    checked = checked_view_gene_lists(given, ("gene_expression", "mutations"))
    assert checked == given
    assert checked is not given
    assert checked_view_gene_lists(None, ("gene_expression",)) == {}
    with pytest.raises(InvalidInputError, match=r"unknown views \['methylation'\]"):
        checked_view_gene_lists({"methylation": None}, ("gene_expression",))


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
