"""Provenance record for the gene-list CSVs shipped next to this module.

This is a maintenance script, **not** part of the public API. Nothing in the package
imports it: the name starts with an underscore, so neither the built-in component
discovery in :mod:`drevalpy.registry._builtins` nor the recursive API docs pick it up.
It exists so the origin of every shipped CSV stays reproducible.

It replaces the ``make_gene_lists.ipynb`` notebook that used to live here. That notebook
also intersected the retired v1 toy datasets, which were synthetic subsets of the real
screens and no longer exist, so only registered datasets are used now (see
``drevalpy/data/datasets/available_datasets.json``). Dropping them leaves ``CCLE`` as the
sole proteomics source and leaves every other intersection over the real screens it
already covered, so the shipped CSVs stay reproducible.

Inputs
------
Per-dataset omics tables, as ``<data-path>/<DATASET>/<omic>.csv`` with a
``cell_line_name`` index and one column per gene. These raw downloads are not part of the
repository; point ``--data-path`` at a local data directory.

Three curated lists are read back from this directory rather than derived:
``landmark_genes.csv``, ``drug_target_genes_all_drugs.csv`` and
``gene_list_paccmann_network_prop.csv``. The original notebook did not record where they
came from ("Todo: how did we get here?"), so this script does not claim to reproduce them
either - it only documents that everything else is derived from them.

Outputs
-------
Written into ``--output-dir`` (this directory by default):

* ``gene_expression_intersection.csv``, ``mutations_intersection.csv``,
  ``methylation_intersection.csv``, ``proteomics_intersection.csv``,
  ``copy_number_variation_gistic_intersection.csv`` - genes present in that omic across
  every dataset listed in :data:`OMIC_DATASETS`.
* ``landmark_genes_proteomics.csv``, ``drug_target_genes_all_drugs_proteomics.csv``,
  ``gene_list_paccmann_network_prop_proteomics.csv`` - each curated list restricted to the
  proteomics intersection.
* ``landmark_genes_reduced.csv``, ``drug_target_genes_reduced.csv``,
  ``gene_list_paccmann_network_prop_reduced.csv`` - each curated list restricted to the
  genes shared by the copy-number, expression, mutation and proteomics intersections.

Two quirks of the shipped files are kept on purpose, so re-running this script produces
the same layout: the ``*_intersection.csv`` files carry pandas' default integer index
column while the derived lists are written with ``index=False``, and row order follows
Python set iteration order, so it is not stable between runs (the set of symbols is).

Usage
-----
``python _make_gene_lists.py --data-path /path/to/data``
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable

import pandas as pd
from upath import UPath

#: Every registered dataset, in the order the notebook listed them.
ALL_DATASETS: tuple[str, ...] = ("BeatAML2", "CCLE", "CTRPv1", "CTRPv2", "GDSC1", "GDSC2", "PDX_Bruna")

#: Datasets that carry each omic. Only these are intersected for that omic, because a
#: dataset without the measurement would empty the intersection.
OMIC_DATASETS: dict[str, tuple[str, ...]] = {
    "copy_number_variation_gistic": ("CCLE", "CTRPv1", "CTRPv2", "GDSC1", "GDSC2", "PDX_Bruna"),
    "gene_expression": ALL_DATASETS,
    "methylation": ("CCLE", "CTRPv1", "CTRPv2", "GDSC1", "GDSC2"),
    "mutations": ("CCLE", "CTRPv1", "CTRPv2", "GDSC1", "GDSC2"),
    # Proteomics is CCLE-only for now; add datasets here once more ship the measurement.
    "proteomics": ("CCLE",),
}

#: Curated input list -> (proteomics-restricted output stem, reduced output stem). The
#: reduced drug-target file drops the ``_all_drugs`` part of its name; that asymmetry is
#: inherited from the shipped files and must be preserved.
DERIVED_STEMS: dict[str, tuple[str, str]] = {
    "landmark_genes": ("landmark_genes_proteomics", "landmark_genes_reduced"),
    "drug_target_genes_all_drugs": ("drug_target_genes_all_drugs_proteomics", "drug_target_genes_reduced"),
    "gene_list_paccmann_network_prop": (
        "gene_list_paccmann_network_prop_proteomics",
        "gene_list_paccmann_network_prop_reduced",
    ),
}

#: Omics whose intersections define the "reduced" gene universe. Methylation is excluded:
#: its columns are genomic ranges, not gene symbols.
REDUCED_OMICS: tuple[str, ...] = ("copy_number_variation_gistic", "gene_expression", "mutations", "proteomics")


def read_omic_genes(dataset: str, data_path: UPath, omic: str) -> set[str]:
    """Return the gene columns of one dataset's omic table.

    :param dataset: Registered dataset name, e.g. ``"GDSC1"``.
    :param data_path: Directory holding one subdirectory per dataset.
    :param omic: Omic file stem, e.g. ``"gene_expression"``.
    :returns: Gene symbols (or, for methylation, genomic ranges) measured in that dataset.
    """
    frame = pd.read_csv(data_path / dataset / f"{omic}.csv", index_col="cell_line_name")
    return {str(column) for column in frame.columns if column != "cellosaurus_id"}


def gene_intersection(datasets: Iterable[str], data_path: UPath, omic: str) -> set[str]:
    """Intersect the genes measured for one omic across several datasets.

    :param datasets: Registered dataset names to intersect.
    :param data_path: Directory holding one subdirectory per dataset.
    :param omic: Omic file stem, e.g. ``"mutations"``.
    :returns: Genes present in every given dataset.
    :raises ValueError: If no dataset was given.
    """
    shared: set[str] | None = None
    for dataset in datasets:
        print(f"Processing {dataset} ({omic})...")
        genes = read_omic_genes(dataset, data_path, omic)
        shared = genes if shared is None else shared & genes
    if shared is None:
        msg = f"No datasets given for omic {omic!r}"
        raise ValueError(msg)
    return shared


def write_symbols(symbols: Iterable[str], path: UPath, *, keep_index: bool) -> None:
    """Write gene symbols as a one-column ``Symbol`` CSV.

    :param symbols: Gene symbols to write.
    :param path: Destination CSV path.
    :param keep_index: Whether to keep pandas' integer index column, as the shipped
        ``*_intersection.csv`` files do.
    """
    pd.DataFrame({"Symbol": list(symbols)}).to_csv(path, index=keep_index)


def read_curated_symbols(stem: str, gene_lists_dir: UPath) -> set[str]:
    """Read the ``Symbol`` column of a curated gene list checked into this directory.

    :param stem: Filename stem without ``.csv``.
    :param gene_lists_dir: Directory holding the curated lists.
    :returns: Curated gene symbols.
    """
    return {str(symbol) for symbol in pd.read_csv(gene_lists_dir / f"{stem}.csv")["Symbol"]}


def build_intersections(data_path: UPath, output_dir: UPath) -> dict[str, set[str]]:
    """Write one ``<omic>_intersection.csv`` per omic and return the gene sets.

    :param data_path: Directory holding one subdirectory per dataset.
    :param output_dir: Directory the CSVs are written to.
    :returns: Mapping of omic name to the intersected gene set.
    """
    intersections: dict[str, set[str]] = {}
    for omic, datasets in OMIC_DATASETS.items():
        genes = gene_intersection(datasets, data_path, omic)
        intersections[omic] = genes
        write_symbols(genes, output_dir / f"{omic}_intersection.csv", keep_index=True)
        print(f"{omic}: {len(genes)} genes shared by {', '.join(datasets)}")
    return intersections


def build_derived_lists(intersections: dict[str, set[str]], gene_lists_dir: UPath, output_dir: UPath) -> None:
    """Restrict each curated list to the proteomics and reduced gene universes.

    :param intersections: Per-omic gene sets from :func:`build_intersections`.
    :param gene_lists_dir: Directory holding the curated input lists.
    :param output_dir: Directory the derived CSVs are written to.
    """
    proteomics = intersections["proteomics"]
    reduced_universe = set.intersection(*(intersections[omic] for omic in REDUCED_OMICS))
    print(f"reduced universe: {len(reduced_universe)} genes shared by {', '.join(REDUCED_OMICS)}")

    for stem, (proteomics_stem, reduced_stem) in DERIVED_STEMS.items():
        curated = read_curated_symbols(stem, gene_lists_dir)
        write_symbols(curated & proteomics, output_dir / f"{proteomics_stem}.csv", keep_index=False)
        write_symbols(curated & reduced_universe, output_dir / f"{reduced_stem}.csv", keep_index=False)
        print(f"{stem}: {len(curated & proteomics)} proteomics, {len(curated & reduced_universe)} reduced")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: Argument list, defaulting to ``sys.argv[1:]``.
    :returns: Parsed arguments with ``data_path`` and ``output_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-path",
        default="data",
        help="Directory holding one subdirectory of raw omics CSVs per dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the gene lists (default: the directory of this script).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Regenerate every gene list shipped in this directory.

    :param argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    gene_lists_dir = UPath(__file__).resolve().parent
    output_dir = UPath(args.output_dir) if args.output_dir else gene_lists_dir
    data_path = UPath(args.data_path)

    intersections = build_intersections(data_path, output_dir)
    build_derived_lists(intersections, gene_lists_dir, output_dir)


if __name__ == "__main__":
    main()
