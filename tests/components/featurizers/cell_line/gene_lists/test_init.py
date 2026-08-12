"""Tests for gene-list resolution helpers."""

from __future__ import annotations

import pandas as pd
import pytest
from upath import UPath

from drevalpy.components.featurizers.cell_line.gene_lists import (
    GENE_LISTS_DIR,
    gene_names_from_list_csv,
    resolve_gene_list_path,
)


def test_gene_names_from_symbol_column(tmp_path: UPath) -> None:
    path = tmp_path / "genes.csv"
    pd.DataFrame({"Symbol": ["A", "B", "C"]}).to_csv(path, index=False)
    assert gene_names_from_list_csv(path) == ["A", "B", "C"]


def test_gene_names_from_gene_name_column(tmp_path: UPath) -> None:
    path = tmp_path / "genes.csv"
    pd.DataFrame({"gene_name": ["X", "Y"]}).to_csv(path, index=False)
    assert gene_names_from_list_csv(path) == ["X", "Y"]


def test_gene_names_rejects_unknown_columns(tmp_path: UPath) -> None:
    path = tmp_path / "genes.csv"
    pd.DataFrame({"other": ["A"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="recognized gene-name column"):
        gene_names_from_list_csv(path)


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
    with pytest.raises(FileNotFoundError, match="Available gene lists: .*landmark_genes"):
        resolve_gene_list_path("not_a_gene_list")
