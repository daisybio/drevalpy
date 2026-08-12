"""Gene-list CSVs shipped with drevalpy, plus helpers to resolve and parse them."""

from __future__ import annotations

import pandas as pd
from upath import UPath as Path

__all__ = ["GENE_LISTS_DIR", "gene_names_from_list_csv", "resolve_gene_list_path"]

_GENE_NAME_COLUMNS = ("Symbol", "gene_name", "symbol", "Gene", "gene")

# The CSVs live next to this module and are shipped as package data, so featurizers never
# depend on a downloaded ``meta`` bundle or any other external location.
GENE_LISTS_DIR = Path(__file__).resolve().parent


def resolve_gene_list_path(
    gene_list_stem: str,
) -> Path:
    """Resolve ``{stem}.csv`` among the gene lists shipped with the package.

    :param gene_list_stem: Gene-list filename stem without ``.csv``.
    :returns: Path to the resolved gene-list CSV.
    :raises FileNotFoundError: If no matching gene-list file exists.
    """
    path = GENE_LISTS_DIR / f"{gene_list_stem}.csv"
    if path.is_file():
        return path
    available = ", ".join(sorted(candidate.stem for candidate in GENE_LISTS_DIR.glob("*.csv")))
    msg = f"Gene list {gene_list_stem!r} not found in {GENE_LISTS_DIR}. Available gene lists: {available}"
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
