"""Helpers for resolving and parsing gene-list CSVs used by feature loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_GENE_NAME_COLUMNS = ("Symbol", "gene_name", "symbol", "Gene", "gene")


def _candidate_gene_lists_dirs() -> list[Path]:
    """Return directories that may contain package/repo gene-list CSVs.

    :returns: Candidate gene-list directories in search order.
    """
    package_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    return [
        repo_root / "data" / "meta" / "gene_lists",
        package_root / "assets" / "gene_lists",
        Path(__file__).resolve().parent / "meta" / "gene_lists",
    ]


def default_gene_lists_dir() -> Path:
    """Return the first existing package/repo gene-lists directory.

    :returns: First existing candidate directory, or the primary default path.
    """
    for path in _candidate_gene_lists_dirs():
        if path.is_dir():
            return path
    return _candidate_gene_lists_dirs()[0]


def resolve_gene_list_path(
    gene_list_stem: str,
    *,
    data_path: str | Path | None = None,
) -> Path:
    """Resolve ``{stem}.csv`` under ``data_path/meta/gene_lists`` or package defaults.

    :param gene_list_stem: Gene-list filename stem without ``.csv``.
    :param data_path: Optional dataset root that may contain ``meta/gene_lists``.
    :returns: Path to the resolved gene-list CSV.
    :raises FileNotFoundError: If no matching gene-list file exists.
    """
    candidates: list[Path] = []
    if data_path is not None:
        candidates.append(Path(data_path) / "meta" / "gene_lists" / f"{gene_list_stem}.csv")
    candidates.extend(directory / f"{gene_list_stem}.csv" for directory in _candidate_gene_lists_dirs())
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    msg = f"Gene list {gene_list_stem!r} not found. Searched: {searched}"
    raise FileNotFoundError(msg)


def gene_names_from_list_csv(path: Path | str) -> list[str]:
    """Return ordered gene symbols from a gene-list CSV.

    Accepts common column names (``Symbol``, ``gene_name``, …).

    :param path: Path to a gene-list CSV.
    :returns: Ordered gene symbol strings.
    :raises ValueError: If the CSV has no recognized gene-name column.
    """
    gene_info = pd.read_csv(path)
    for column in _GENE_NAME_COLUMNS:
        if column in gene_info.columns:
            return [str(value) for value in gene_info[column].tolist()]
    msg = (
        f"Gene list {path} has no recognized gene-name column; "
        f"expected one of {list(_GENE_NAME_COLUMNS)}, got {list(gene_info.columns)}"
    )
    raise ValueError(msg)
