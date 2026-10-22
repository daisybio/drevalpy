"""
Builds the gene lists behind the TCGA autoencoder gene space (``tcga_autoencoder_genes_available``, 11,883 genes).

Why this gene space exists: a model that is trained on gene expression and a model whose gene encoder is pretrained on
TCGA tumours can only be compared if both see the same genes, otherwise the comparison mixes the effect of the
pretraining with the effect of the gene set. The list is the largest gene set that fits all of these constraints:
measured in CTRPv2 and GDSC1 (so every response dataset can use it), covered by the Ion AmpliSeq panel (so a new
sample sequenced with that panel has every feature), expressed above noise level (so the autoencoder does not learn
from unexpressed genes), and mappable to TCGA (so the same genes exist in the pretraining data). Expression noise is
removed, but no top-variance selection is made, because that would remove driver genes; the bottleneck of the
autoencoder does the dimensionality reduction.
"""

import argparse
import io
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

#: Columns of a gene expression file that are no genes.
ID_COLUMNS = {"cell_line_name", "cellosaurus_id"}
#: Noise filter: drop a gene if mean(log2(x + 1)) or the variance of log2(x + 1) is below these values.
MIN_MEAN = 0.5
MIN_VARIANCE = 1e-3
#: Public gene content of the Ion AmpliSeq Transcriptome panel (AmpliSeq for Illumina RNA manifest, dated, anonymous).
MANIFEST_URL = (
    "https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/"
    "ampliseq-for-illumina/ampliseq-for-illumina-transcriptome-rna-manifest-20190313.zip"
)
#: TCGA gene annotation (GENCODE v23 probemap of the Xena Toil hub), columns ``id`` and ``gene``.
PROBEMAP_URL = (
    "https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/probeMap%2Fgencode.v23.annotation.gene.probemap"
)
#: Clinically actionable driver genes that the noise filter never drops, in addition to the landmark genes.
CLINICAL_DRIVER_GENES = {
    "ALK", "BRAF", "CTNNB1", "DDR2", "ERBB4", "ESR1", "FBXW7", "FGFR1", "FGFR3", "FLT3", "H3-3A", "HRAS", "IDH1",
    "IDH2", "JAK2", "KRAS", "MAP2K1", "MDM2", "MET", "NF1", "PDGFRA", "PTEN", "RET", "ROS1", "SMAD4", "STK11",
}  # fmt: skip
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


def parse_manifest(text: str) -> set[str]:
    """
    Read the gene symbols of the ``[Targets]`` section of an AmpliSeq for Illumina RNA manifest.

    :param text: content of the manifest (tab separated, sections in square brackets)
    :returns: gene symbols (column ``Gene``)
    """
    lines = text.splitlines()
    start = lines.index("[Targets]")
    column = lines[start + 1].split("\t").index("Gene")
    genes = set()
    for line in lines[start + 2 :]:  # noqa: E203
        if line.startswith("["):
            break
        fields = line.split("\t")
        if len(fields) > column and fields[column].strip():
            genes.add(fields[column].strip())
    return genes


def download_manifest_genes(url: str = MANIFEST_URL, timeout: int = 120) -> set[str]:
    """
    Download the public manifest of the panel and return its gene symbols.

    :param url: zip file with one manifest text file
    :param timeout: seconds
    :returns: gene symbols
    """
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    return parse_manifest(archive.read(archive.namelist()[0]).decode("utf-8", "replace"))


def download_probemap_genes(url: str = PROBEMAP_URL, timeout: int = 120) -> set[str]:
    """
    Download the TCGA gene annotation and return its gene symbols.

    :param url: tab-separated probemap with a ``gene`` column
    :param timeout: seconds
    :returns: gene symbols
    """
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return set(pd.read_csv(io.BytesIO(response.read()), sep="\t", usecols=["gene"], dtype=str)["gene"])


def read_symbols(path: Path) -> set[str]:
    """
    Reads a gene list with a ``Symbol`` column.

    :param path: gene list csv
    :returns: gene symbols
    """
    return set(pd.read_csv(path, dtype=str)["Symbol"])


def noise_filter(expression: pd.DataFrame, protected: set[str]) -> set[str]:
    """
    Selects the genes that are expressed above noise level.

    A gene is dropped if the mean of log2(expression + 1) over the cell lines is below ``MIN_MEAN`` or its variance is
    below ``MIN_VARIANCE``, unless it is protected (driver and landmark genes).

    :param expression: cell lines x genes, raw expression
    :param protected: genes that are never dropped
    :returns: the genes that stay
    """
    logged = np.log2(expression.to_numpy(dtype=float) + 1)
    mean, variance = pd.Series(logged.mean(axis=0), expression.columns), pd.Series(
        logged.var(axis=0), expression.columns
    )
    noisy = (mean < MIN_MEAN) | (variance < MIN_VARIANCE)
    return set(expression.columns[~noisy]) | (protected & set(expression.columns))


def build_chain(
    expression: list[set[str]], panel: set[str], expressed: Callable[[set[str]], set[str]], tcga_symbols: set[str]
) -> dict[str, set[str]]:
    """
    Builds the four gene lists, each a subset of the one before.

    1. genes in the expression data of all datasets;
    2. of those, the genes on the sequencing panel the response data were measured with;
    3. of those, the genes that are expressed above noise level (``expressed`` applies the noise filter);
    4. of those, the genes that have a locus in the TCGA gene annotation, since the others cannot be mapped to TCGA.

    :param expression: gene symbols of the expression data of every dataset
    :param panel: gene symbols of the sequencing panel
    :param expressed: function that keeps the expressed genes of a gene set
    :param tcga_symbols: gene symbols that have a locus in the TCGA gene annotation (GENCODE v23)
    :returns: step name -> gene symbols
    :raises ValueError: if no dataset is given
    """
    if not expression:
        raise ValueError("At least one gene expression file is needed.")
    chain = {STEP_FILES[0]: set.intersection(*expression)}
    chain[STEP_FILES[1]] = chain[STEP_FILES[0]] & panel
    chain[STEP_FILES[2]] = expressed(chain[STEP_FILES[1]])
    chain[STEP_FILES[3]] = chain[STEP_FILES[2]] & tcga_symbols
    return chain


def main():
    """Read the inputs and write the four gene lists to <data_path>/meta/gene_lists."""
    parser = argparse.ArgumentParser(description="Build the gene lists of the TCGA autoencoder gene space.")
    parser.add_argument("--data_path", type=str, default="data", help="Data directory with the dataset folders.")
    parser.add_argument("--datasets", type=str, nargs="+", default=["CTRPv2", "GDSC1"], help="Expression data.")
    parser.add_argument(
        "--panel",
        type=str,
        default=None,
        help="Ion AmpliSeq Transcriptome panel file (BED). Default: download the public Illumina manifest.",
    )
    parser.add_argument("--panel_symbol_column", type=int, default=6, help="Zero-based gene symbol column.")
    parser.add_argument(
        "--protect",
        type=str,
        default=None,
        help="Gene list (Symbol) that the noise filter never drops. Default: the landmark genes of the metadata "
        "bundle plus the clinical driver genes.",
    )
    parser.add_argument("--noise_dataset", type=str, default="CTRPv2", help="Dataset whose expression defines noise.")
    parser.add_argument(
        "--probemap",
        type=str,
        default=None,
        help="TCGA gencode.v23.probemap.tsv (column gene). Downloaded from the Xena Toil hub if not given.",
    )
    parser.add_argument("--output", type=str, default=None, help="Directory (default: <data_path>/meta/gene_lists).")
    args = parser.parse_args()
    data_path = Path(args.data_path)
    output = Path(args.output) if args.output else data_path / "meta" / "gene_lists"
    if args.protect:
        protected = read_symbols(Path(args.protect))
    else:
        protected = read_symbols(data_path / "meta" / "gene_lists" / "landmark_genes.csv") | CLINICAL_DRIVER_GENES

    def expressed(genes: set[str]) -> set[str]:
        columns = sorted(genes)
        expression = pd.read_csv(data_path / args.noise_dataset / "gene_expression.csv", usecols=columns)
        return noise_filter(expression, protected)

    chain = build_chain(
        [expression_genes(data_path / name / "gene_expression.csv") for name in args.datasets],
        read_panel_genes(Path(args.panel), args.panel_symbol_column) if args.panel else download_manifest_genes(),
        expressed,
        (
            set(pd.read_csv(args.probemap, sep="\t", usecols=["gene"], dtype=str)["gene"])
            if args.probemap
            else download_probemap_genes()
        ),
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
