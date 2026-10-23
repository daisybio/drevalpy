"""Gene-list CSVs shipped with drevalpy, plus helpers to resolve, parse and apply them.

The CSVs in this directory are the only source of gene lists: datasets never carry their own,
so a list name means the same genes on every dataset. :func:`resolve_gene_list` reads one by
name and :func:`select_genes` applies it to a view, raising when the view lacks any of its
genes. ``GENE_LISTS_BY_VIEW`` records which view each list belongs to.

See the maintenance scripts ``_make_gene_lists.py`` and ``_make_autoencoder_gene_list.py`` in
this directory for how each CSV was produced.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from upath import UPath as Path

from drevalpy.errors import InvalidInputError, UnknownNameError

if TYPE_CHECKING:
    from drevalpy.types.data.feature_source import FeatureSource

__all__ = [
    "GENE_LISTS_BY_VIEW",
    "GENE_LISTS_DIR",
    "checked_view_gene_lists",
    "gene_names_from_list_csv",
    "resolve_gene_list",
    "resolve_gene_list_path",
    "select_genes",
    "view_gene_union",
]

_GENE_NAME_COLUMNS = ("Symbol", "gene_name", "symbol", "Gene", "gene")
#: How many missing genes the error message names.
_MISSING_PREVIEW = 5

# The CSVs live next to this module and are shipped as package data, so featurizers never
# depend on a downloaded ``meta`` bundle or any other external location.
GENE_LISTS_DIR = Path(__file__).resolve().parent

#: The shipped lists that apply to each sample view, in the order :func:`view_gene_union`
#: concatenates them. Every shipped CSV appears at least once; a test fails on a new CSV that
#: is not assigned here, because the synthetic datasets and the curated toy set derive the genes
#: of each view from this table.
GENE_LISTS_BY_VIEW: Final[Mapping[str, tuple[str, ...]]] = {
    "gene_expression": (
        "landmark_genes",
        "landmark_genes_reduced",
        "gene_expression_intersection",
        "gene_list_paccmann_network_prop",
        "gene_list_paccmann_network_prop_reduced",
        "drug_target_genes_all_drugs",
        "drug_target_genes_reduced",
        "tcga_autoencoder_genes_available",
    ),
    "mutations": ("mutations_intersection", "drug_target_genes_all_drugs", "drug_target_genes_reduced"),
    "copy_number_variation_gistic": (
        "copy_number_variation_gistic_intersection",
        "drug_target_genes_all_drugs",
        "drug_target_genes_reduced",
    ),
    "methylation": ("methylation_intersection",),
    "proteomics": (
        "proteomics_intersection",
        "landmark_genes_proteomics",
        "drug_target_genes_all_drugs_proteomics",
        "gene_list_paccmann_network_prop_proteomics",
    ),
}


def resolve_gene_list_path(
    gene_list_stem: str,
) -> Path:
    """Resolve ``{stem}.csv`` among the gene lists shipped with the package.

    :param gene_list_stem: Gene-list filename stem without ``.csv``.
    :returns: Path to the resolved gene-list CSV.
    :raises UnknownNameError: If no matching gene-list file exists; the message lists the shipped ones.
    """
    path = GENE_LISTS_DIR / f"{gene_list_stem}.csv"
    if path.is_file():
        return path
    available = ", ".join(sorted(candidate.stem for candidate in GENE_LISTS_DIR.glob("*.csv")))
    msg = f"Gene list {gene_list_stem!r} not found in {GENE_LISTS_DIR}. Available gene lists: {available}"
    raise UnknownNameError(msg)


def gene_names_from_list_csv(path: Path | str) -> list[str]:
    """Return ordered gene symbols from a gene-list CSV.

    Accepts common column names (``Symbol``, ``gene_name``, …).

    :param path: Path to a gene-list CSV.
    :returns: Ordered gene symbol strings.
    :raises InvalidInputError: If the CSV has no recognized gene-name column.
    """
    import pandas as pd

    gene_info = pd.read_csv(path)
    for column in _GENE_NAME_COLUMNS:
        if column in gene_info.columns:
            return [str(value) for value in gene_info[column].tolist()]
    msg = (
        f"Gene list {path} has no recognized gene-name column; "
        f"expected one of {list(_GENE_NAME_COLUMNS)}, got {list(gene_info.columns)}"
    )
    raise InvalidInputError(msg)


def resolve_gene_list(name: str) -> tuple[str, ...]:
    """Return the genes of a shipped gene list, in file order and without repeats.

    :param name: Gene-list name, the CSV filename stem such as ``landmark_genes``.
    :returns: Ordered, de-duplicated gene symbols.
    """
    return tuple(dict.fromkeys(gene_names_from_list_csv(resolve_gene_list_path(name))))


def select_genes(source: FeatureSource, view: str, name: str) -> tuple[str, ...]:
    """Return the genes of list *name*, in list order, after checking *source* holds all of them.

    :param source: Feature source providing the view's feature names.
    :param view: View to read, such as ``gene_expression``.
    :param name: Gene-list name, resolved by :func:`resolve_gene_list`.
    :returns: The list's genes in list order; the column order a featurizer reads the view in.
    :raises InvalidInputError: If the view has no feature names, or lacks genes of the list.
    """
    names = source.get_feature_names(view)
    if names is None:
        msg = f"FeatureSource has no feature names for view {view!r}"
        raise InvalidInputError(msg)
    genes = resolve_gene_list(name)
    available = {str(gene) for gene in names}
    missing = [gene for gene in genes if gene not in available]
    if missing:
        preview = ", ".join(missing[:_MISSING_PREVIEW]) + (", ..." if len(missing) > _MISSING_PREVIEW else "")
        msg = (
            f"view {view!r} lacks {len(missing)} of {len(genes)} genes in {name!r} (missing: {preview}). "
            "Pass gene_list=None to use every gene of the view."
        )
        raise InvalidInputError(msg)
    return genes


def view_gene_union(view: str) -> tuple[str, ...]:
    """Return every gene the shipped lists of *view* name, each once, in first-appearance order.

    :param view: Sample view, a key of :data:`GENE_LISTS_BY_VIEW`.
    :returns: Ordered, de-duplicated genes; empty for a view without shipped lists.
    """
    genes = (gene for name in GENE_LISTS_BY_VIEW.get(view, ()) for gene in resolve_gene_list(name))
    return tuple(dict.fromkeys(genes))


def checked_view_gene_lists(
    gene_lists: Mapping[str, str | None] | None, views: tuple[str, ...]
) -> dict[str, str | None]:
    """Copy a per-view gene-list mapping after checking that it only names known views.

    :param gene_lists: Gene-list name per view, or ``None`` for no lists.
    :param views: The views the featurizer reads.
    :returns: A new mapping with the same entries.
    :raises InvalidInputError: If a key is not one of *views*.
    """
    unknown = sorted(set(gene_lists or {}) - set(views))
    if unknown:
        msg = f"gene_lists names unknown views {unknown}; choose from {list(views)}"
        raise InvalidInputError(msg)
    return dict(gene_lists or {})
