"""Tests for gene-list resolution helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from drevalpy.data.gene_lists import gene_names_from_list_csv, resolve_gene_list_path


def test_gene_names_from_symbol_column(tmp_path: Path) -> None:
    path = tmp_path / "genes.csv"
    pd.DataFrame({"Symbol": ["A", "B", "C"]}).to_csv(path, index=False)
    assert gene_names_from_list_csv(path) == ["A", "B", "C"]


def test_gene_names_from_gene_name_column(tmp_path: Path) -> None:
    path = tmp_path / "genes.csv"
    pd.DataFrame({"gene_name": ["X", "Y"]}).to_csv(path, index=False)
    assert gene_names_from_list_csv(path) == ["X", "Y"]


def test_gene_names_rejects_unknown_columns(tmp_path: Path) -> None:
    path = tmp_path / "genes.csv"
    pd.DataFrame({"other": ["A"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="recognized gene-name column"):
        gene_names_from_list_csv(path)


def test_resolve_gene_list_path_uses_data_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gene_dir = tmp_path / "meta" / "gene_lists"
    gene_dir.mkdir(parents=True)
    path = gene_dir / "custom_genes.csv"
    pd.DataFrame({"Symbol": ["G1"]}).to_csv(path, index=False)
    monkeypatch.setenv("DREVALPY_CACHE_DIR", str(tmp_path))
    assert resolve_gene_list_path("custom_genes") == path
