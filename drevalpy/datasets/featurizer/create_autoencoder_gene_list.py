"""Builds the gene lists behind the TCGA autoencoder gene space (``tcga_autoencoder_genes_available``, 11,883 genes)."""

import argparse
from pathlib import Path

import pandas as pd

#: Columns of a gene expression file that are no genes.
ID_COLUMNS = {"cell_line_name", "cellosaurus_id"}
#: Output names, one per step of the chain, in the order they are built.
STEP_FILES = [
    "ctrpv2_gdsc1_intersection",
    "ctrpv2_gdsc1_iontorrent_intersection",
    "autoencoder_expressed_genes",
    "tcga_autoencoder_genes_available",
]


def expression_genes(path: Path) -> set[str]:
    """
    Reads the gene symbols (the column names) of a gene expression file.

    :param path: gene_expression.csv of a dataset
    :returns: gene symbols
    """
    return set(pd.read_csv(path, nrows=0).columns) - ID_COLUMNS


def read_panel_genes(path: Path, symbol_column: int = 6) -> set[str]:
    """
    Reads the gene symbols of the Ion AmpliSeq Transcriptome Human Gene Expression Panel.

    The panel file is the public BED file of the panel: semicolon separated, no header. Its spike-in controls (ERCC)
    are harmless, because no expression file has them as a column.

    :param path: panel file
    :param symbol_column: zero-based column with the gene symbol
    :returns: gene symbols
    """
    return set(pd.read_csv(path, sep=";", header=None, dtype=str)[symbol_column].str.strip())


def read_symbols(path: Path) -> set[str]:
    """
    Reads a gene list with a ``Symbol`` column.

    :param path: gene list csv
    :returns: gene symbols
    """
    return set(pd.read_csv(path, dtype=str)["Symbol"])


def build_chain(
    expression: list[set[str]], panel: set[str], vocabulary: set[str], tcga_symbols: set[str]
) -> dict[str, set[str]]:
    """
    Builds the four gene lists, each a subset of the one before.

    1. genes in the expression data of all datasets;
    2. of those, the genes on the sequencing panel the response data were measured with;
    3. of those, the genes in the input vocabulary of the pretrained autoencoder (plain set membership);
    4. of those, the genes that have a locus in the TCGA gene annotation, since the others cannot be mapped to TCGA.

    :param expression: gene symbols of the expression data of every dataset
    :param panel: gene symbols of the sequencing panel
    :param vocabulary: gene symbols the pretrained autoencoder was built on
    :param tcga_symbols: gene symbols that have a locus in the TCGA gene annotation (GENCODE v23)
    :returns: step name -> gene symbols
    :raises ValueError: if no dataset is given
    """
    if not expression:
        raise ValueError("At least one gene expression file is needed.")
    chain = {STEP_FILES[0]: set.intersection(*expression)}
    chain[STEP_FILES[1]] = chain[STEP_FILES[0]] & panel
    chain[STEP_FILES[2]] = chain[STEP_FILES[1]] & vocabulary
    chain[STEP_FILES[3]] = chain[STEP_FILES[2]] & tcga_symbols
    return chain


def main():
    """Read the inputs and write the four gene lists to <data_path>/meta/gene_lists."""
    parser = argparse.ArgumentParser(description="Build the gene lists of the TCGA autoencoder gene space.")
    parser.add_argument("--data_path", type=str, default="data", help="Data directory with the dataset folders.")
    parser.add_argument("--datasets", type=str, nargs="+", default=["CTRPv2", "GDSC1"], help="Expression data.")
    parser.add_argument("--panel", type=str, required=True, help="Ion AmpliSeq Transcriptome panel file (BED).")
    parser.add_argument("--panel_symbol_column", type=int, default=6, help="Zero-based gene symbol column.")
    parser.add_argument(
        "--vocabulary", type=str, required=True, help="Gene list (Symbol) of the pretrained autoencoder."
    )
    parser.add_argument("--probemap", type=str, required=True, help="TCGA gencode.v23.probemap.tsv (column gene).")
    parser.add_argument("--output", type=str, default=None, help="Directory (default: <data_path>/meta/gene_lists).")
    args = parser.parse_args()
    data_path = Path(args.data_path)
    output = Path(args.output) if args.output else data_path / "meta" / "gene_lists"
    chain = build_chain(
        [expression_genes(data_path / name / "gene_expression.csv") for name in args.datasets],
        read_panel_genes(Path(args.panel), args.panel_symbol_column),
        read_symbols(Path(args.vocabulary)),
        set(pd.read_csv(args.probemap, sep="\t", usecols=["gene"], dtype=str)["gene"]),
    )
    output.mkdir(parents=True, exist_ok=True)
    previous: set[str] = set()
    for name, genes in chain.items():
        pd.DataFrame({"Symbol": sorted(genes)}).to_csv(output / f"{name}.csv", index=False)
        dropped = sorted(previous - genes) if previous else []
        print(f"{name}: {len(genes)} genes" + (f" (dropped {len(dropped)}: {dropped[:5]})" if dropped else ""))
        previous = genes


if __name__ == "__main__":
    main()
