"""
Builds the gene list of the TCGA autoencoder gene space (``tcga_autoencoder_genes_available``).

Why this gene space exists: a model that is trained on gene expression and a model whose gene encoder is pretrained on
TCGA tumours can only be compared if both see the same genes, otherwise the comparison mixes the effect of the
pretraining with the effect of the gene set. The list is the largest gene set that fits all of these constraints:
measured in all given expression datasets (so every response dataset can use it), covered by the Ion AmpliSeq panel (so
a new sample sequenced with that panel has every feature), expressed above noise level (so the autoencoder does not
learn from unexpressed genes), and mappable to TCGA (so the same genes exist in the pretraining data). Expression noise
is removed, but no top-variance selection is made, because that would remove driver genes; the bottleneck of the
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
#: Datasets whose expression is linear TPM (the others are already log scaled), as in DIPK.
TPM_DATASETS = {"CTRPv1", "CTRPv2", "CCLE"}
#: Raw TPM reaches values in the thousands; a maximum below this means the data are already log scaled.
MIN_RAW_MAXIMUM = 100.0
#: Name of the final gene list (a file in ``<data_path>/meta/gene_lists``).
GENE_LIST_NAME = "tcga_autoencoder_genes_available"
#: Names of the steps of the chain, in the order they are built.
STEPS = ["expression_intersection", "panel_intersection", "expressed_genes", GENE_LIST_NAME]


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
    below ``MIN_VARIANCE``, unless it is protected (driver and landmark genes). The thresholds are meant for linear TPM.

    :param expression: cell lines x genes, raw (linear TPM) expression
    :param protected: genes that are never dropped
    :returns: the genes that stay
    :raises ValueError: if the expression looks log scaled already (maximum below ``MIN_RAW_MAXIMUM``), because the
        thresholds do not apply to it
    """
    if np.nanmax(expression.to_numpy(dtype=float)) < MIN_RAW_MAXIMUM:
        raise ValueError(
            "The noise filter needs linear TPM, but the maximum of this expression is below "
            f"{MIN_RAW_MAXIMUM:g}, so it looks log scaled already. Use a TPM dataset ({sorted(TPM_DATASETS)})."
        )
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
    Builds the gene lists of the four steps, each a subset of the one before.

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
    chain = {STEPS[0]: set.intersection(*expression)}
    chain[STEPS[1]] = chain[STEPS[0]] & panel
    chain[STEPS[2]] = expressed(chain[STEPS[1]])
    chain[STEPS[3]] = chain[STEPS[2]] & tcga_symbols
    return chain


def default_noise_dataset(data_path: Path, datasets: list[str]) -> str | None:
    """
    Choose the dataset whose expression defines noise: a TPM dataset among ``datasets``, else CTRPv2 if available.

    :param data_path: data directory with the dataset folders
    :param datasets: expression datasets of the gene list
    :returns: dataset name, or ``None`` if there is no TPM expression at hand
    """
    for name in datasets:
        if name in TPM_DATASETS:
            return name
    return "CTRPv2" if (data_path / "CTRPv2" / "gene_expression.csv").exists() else None


def create_gene_list(
    data_path: Path,
    datasets: list[str],
    noise_dataset: str | None = None,
    panel: Path | None = None,
    panel_symbol_column: int = 6,
    protect: Path | None = None,
    probemap: Path | None = None,
    output: Path | None = None,
    name: str = GENE_LIST_NAME,
) -> set[str]:
    """
    Build the gene list and write it to ``<output>/<name>.csv``.

    :param data_path: data directory with the dataset folders
    :param datasets: expression datasets; the list holds the genes of all of them
    :param noise_dataset: dataset whose expression defines noise, default see :func:`default_noise_dataset`. Without a
        TPM dataset the noise filter is skipped.
    :param panel: Ion AmpliSeq Transcriptome panel (BED); default: download the public Illumina manifest
    :param panel_symbol_column: zero-based gene symbol column of ``panel``
    :param protect: gene list (Symbol) the noise filter never drops; default: landmark genes plus driver genes
    :param probemap: TCGA gencode.v23.probemap.tsv; default: download it from the Xena Toil hub
    :param output: directory, default ``<data_path>/meta/gene_lists``
    :param name: file name of the list, without extension
    :returns: the genes of the list
    """
    output = output or data_path / "meta" / "gene_lists"
    noise_dataset = noise_dataset or default_noise_dataset(data_path, datasets)
    protected = (
        read_symbols(protect)
        if protect
        else read_symbols(data_path / "meta" / "gene_lists" / "landmark_genes.csv") | CLINICAL_DRIVER_GENES
    )

    def expressed(genes: set[str]) -> set[str]:
        if noise_dataset is None:
            print("No TPM expression dataset found, the noise filter is skipped.")
            return genes
        path = data_path / noise_dataset / "gene_expression.csv"
        judged = genes & expression_genes(path)
        if len(judged) < len(genes):
            print(f"{len(genes) - len(judged)} genes are not in {noise_dataset}; the noise filter keeps them.")
        expression = pd.read_csv(path, usecols=sorted(judged))
        return noise_filter(expression, protected) | (genes - judged)

    chain = build_chain(
        [expression_genes(data_path / dataset / "gene_expression.csv") for dataset in datasets],
        read_panel_genes(panel, panel_symbol_column) if panel else download_manifest_genes(),
        expressed,
        (
            set(pd.read_csv(probemap, sep="\t", usecols=["gene"], dtype=str)["gene"])
            if probemap
            else download_probemap_genes()
        ),
    )
    previous: set[str] = set()
    for step, genes in chain.items():
        dropped = sorted(previous - genes) if previous else []
        print(f"{step}: {len(genes)} genes" + (f" (dropped {len(dropped)}: {dropped[:5]})" if dropped else ""))
        previous = genes
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Symbol": sorted(previous)}).to_csv(output / f"{name}.csv", index=False)
    return previous


def main():
    """Read the inputs and write the gene list to <data_path>/meta/gene_lists."""
    parser = argparse.ArgumentParser(description="Build the gene list of the TCGA autoencoder gene space.")
    parser.add_argument("--data_path", type=str, default="data", help="Data directory with the dataset folders.")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["CTRPv2", "GDSC1", "PDX_Bruna", "BeatAML2"],
        help="Expression datasets; the list holds the genes that all of them have.",
    )
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
    parser.add_argument(
        "--noise_dataset",
        type=str,
        default=None,
        help="Dataset whose linear TPM expression defines noise. Default: a TPM dataset among --datasets, else CTRPv2.",
    )
    parser.add_argument(
        "--probemap",
        type=str,
        default=None,
        help="TCGA gencode.v23.probemap.tsv (column gene). Downloaded from the Xena Toil hub if not given.",
    )
    parser.add_argument("--output", type=str, default=None, help="Directory (default: <data_path>/meta/gene_lists).")
    parser.add_argument("--name", type=str, default=GENE_LIST_NAME, help="File name of the list, without extension.")
    args = parser.parse_args()
    create_gene_list(
        Path(args.data_path),
        args.datasets,
        args.noise_dataset,
        Path(args.panel) if args.panel else None,
        args.panel_symbol_column,
        Path(args.protect) if args.protect else None,
        Path(args.probemap) if args.probemap else None,
        Path(args.output) if args.output else None,
        args.name,
    )


if __name__ == "__main__":
    main()
