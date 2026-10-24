"""Tests for gene-list resolution helpers."""

from __future__ import annotations

import pandas as pd
import pytest
from upath import UPath

from drevalpy.components.featurizers.sample.gene_lists import (
    GENE_LISTS_BY_VIEW,
    GENE_LISTS_DIR,
    checked_view_gene_lists,
    gene_names_from_list_csv,
    resolve_gene_list,
    resolve_gene_list_path,
    select_genes,
    view_gene_union,
)
from drevalpy.errors import InvalidInputError, UnknownNameError
from tests.conftest import MockFeatureSource


@pytest.mark.parametrize(("column", "expected"), [("Symbol", ["A", "B"]), ("gene_name", ["A", "B"]), ("other", None)])
def test_gene_names_read_known_columns_and_reject_unknown(tmp_path: UPath, column: str, expected) -> None:
    path = tmp_path / "genes.csv"
    pd.DataFrame({column: ["A", "B"]}).to_csv(path, index=False)
    if expected is None:
        with pytest.raises(InvalidInputError, match="recognized gene-name column"):
            gene_names_from_list_csv(path)
    else:
        assert gene_names_from_list_csv(path) == expected


def test_resolve_gene_list_path_uses_packaged_csv() -> None:
    path = resolve_gene_list_path("landmark_genes")
    assert path.is_file()
    assert path == GENE_LISTS_DIR / "landmark_genes.csv"


def test_resolve_gene_list_path_ignores_cache_dir(tmp_path: UPath, monkeypatch: pytest.MonkeyPatch) -> None:
    gene_dir = tmp_path / "meta" / "gene_lists"
    gene_dir.mkdir(parents=True)
    pd.DataFrame({"Symbol": ["G1"]}).to_csv(gene_dir / "landmark_genes.csv", index=False)
    monkeypatch.setenv("DREVALPY_CACHE_DIR", str(tmp_path))
    assert resolve_gene_list_path("landmark_genes") == GENE_LISTS_DIR / "landmark_genes.csv"


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
    shipped = {path.stem for path in GENE_LISTS_DIR.glob("*.csv")}
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
