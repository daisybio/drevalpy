"""Build ``tcga_autoencoder_genes_available.parquet``, the gene space of the TCGA-pretrained autoencoder.

This is a maintenance script, **not** part of the public API: nothing imports it and its name
starts with an underscore, so component discovery and the API docs skip it. It needs the
network (two public downloads) and is not run in CI.

Why this gene space exists: a model trained on gene expression and a model whose gene encoder is
pretrained on TCGA tumours can only be compared if both see the same genes, otherwise the
comparison mixes the effect of the pretraining with the effect of the gene set. The list is the
largest gene set that fits all of these constraints:

* measured in the expression view of every given dataset, so every response dataset can use it;
* covered by the Ion AmpliSeq panel, so a new sample sequenced with that panel has every feature
  (the gene content is read from the public AmpliSeq for Illumina RNA manifest);
* expressed above noise level, so the autoencoder does not learn from unexpressed genes;
* mappable to TCGA, so the same genes exist in the pretraining data (the GENCODE v23 probemap
  of the Xena Toil hub).

Expression noise is removed, but no top-variance selection is made, because that would remove
driver genes; the bottleneck of the autoencoder does the dimensionality reduction.

Inputs
------
One ``<DATASET>.h5mu`` per dataset in ``--data-path`` (default: the drevalpy cache directory, where
``drevalpy.load`` puts them). Only the ``gene_expression`` modality is read. The noise dataset must
hold ``log2(TPM + 1)`` values (``expression_unit`` of ``log2_tpm_plus_1``); the thresholds are
defined on that scale. ``landmark_genes.parquet`` next to this script names the protected genes.

Usage
-----
``python _make_autoencoder_gene_list.py [--data-path DIR] [--datasets CTRPv2 GDSC1 PDX_Bruna BeatAML2]``
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from collections.abc import Callable

import numpy as np
import pandas as pd
from upath import UPath

from drevalpy.errors import InvalidInputError

#: Noise filter: drop a gene if the mean or the variance of ``log2(TPM + 1)`` is below these values.
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
#: Datasets whose gene-expression view defines the shared genes.
DEFAULT_DATASETS = ("CTRPv2", "GDSC1", "PDX_Bruna", "BeatAML2")
#: ``expression_unit`` of the views the noise thresholds apply to.
NOISE_UNIT = "log2_tpm_plus_1"
GENE_LIST_NAME = "tcga_autoencoder_genes_available"
#: Names of the steps of the chain, in the order they are built.
STEPS = ("expression_intersection", "panel_intersection", "expressed_genes", GENE_LIST_NAME)
_MODALITY = "gene_expression"


def read_expression(path: UPath, *, with_values: bool) -> tuple[list[str], pd.DataFrame | None, str]:
    """Read the ``gene_expression`` modality of a dataset file.

    :param path: ``<DATASET>.h5mu`` file.
    :param with_values: Whether to return the samples-by-genes table as well.
    :returns: Gene symbols, the table (``None`` without *with_values*) and the ``expression_unit``.
    """
    import mudata as md

    modality = md.read_h5ad(path, _MODALITY, backed=not with_values)
    genes = [str(name) for name in modality.var_names]
    unit = str(modality.uns.get("expression_unit", ""))
    if not with_values:
        modality.file.close()
        return genes, None, unit
    return genes, pd.DataFrame(np.asarray(modality.X, dtype=float), columns=genes), unit


def parse_manifest(text: str) -> set[str]:
    """Read the gene symbols of the ``[Targets]`` section of an AmpliSeq for Illumina RNA manifest.

    :param text: Content of the manifest (tab separated, sections in square brackets).
    :returns: Gene symbols (column ``Gene``).
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


def _download(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def download_manifest_genes(url: str = MANIFEST_URL, timeout: int = 120) -> set[str]:
    """Download the public manifest of the panel and return its gene symbols.

    :param url: Zip file with one manifest text file.
    :param timeout: Seconds.
    :returns: Gene symbols.
    """
    archive = zipfile.ZipFile(io.BytesIO(_download(url, timeout)))
    return parse_manifest(archive.read(archive.namelist()[0]).decode("utf-8", "replace"))


def download_probemap_genes(url: str = PROBEMAP_URL, timeout: int = 120) -> set[str]:
    """Download the TCGA gene annotation and return its gene symbols.

    :param url: Tab-separated probemap with a ``gene`` column.
    :param timeout: Seconds.
    :returns: Gene symbols.
    """
    return set(pd.read_csv(io.BytesIO(_download(url, timeout)), sep="\t", usecols=["gene"], dtype=str)["gene"])


def noise_filter(expression: pd.DataFrame, protected: set[str]) -> set[str]:
    """Select the genes that are expressed above noise level.

    A gene is dropped if its mean or its variance over the samples is below :data:`MIN_MEAN` or
    :data:`MIN_VARIANCE`, unless it is protected (landmark and driver genes).

    :param expression: Samples by genes, in ``log2(TPM + 1)``.
    :param protected: Genes that are never dropped.
    :returns: The genes that stay.
    """
    values = expression.to_numpy(dtype=float)
    mean = pd.Series(values.mean(axis=0), expression.columns)
    variance = pd.Series(values.var(axis=0), expression.columns)
    noisy = (mean < MIN_MEAN) | (variance < MIN_VARIANCE)
    return set(expression.columns[~noisy]) | (protected & set(expression.columns))


def build_chain(
    expression: list[set[str]], panel: set[str], expressed: Callable[[set[str]], set[str]], tcga_symbols: set[str]
) -> dict[str, set[str]]:
    """Build the gene lists of the four steps, each a subset of the one before.

    1. genes in the expression view of all datasets;
    2. of those, the genes on the sequencing panel the response data were measured with;
    3. of those, the genes that are expressed above noise level (*expressed* applies the filter);
    4. of those, the genes that have a locus in the TCGA gene annotation.

    :param expression: Gene symbols of the expression view of every dataset.
    :param panel: Gene symbols of the sequencing panel.
    :param expressed: Function that keeps the expressed genes of a gene set.
    :param tcga_symbols: Gene symbols that have a locus in the TCGA gene annotation (GENCODE v23).
    :returns: Step name to gene symbols.
    :raises InvalidInputError: If no dataset is given.
    """
    if not expression:
        msg = "At least one gene-expression view is needed."
        raise InvalidInputError(msg)
    chain = {STEPS[0]: set.intersection(*expression)}
    chain[STEPS[1]] = chain[STEPS[0]] & panel
    chain[STEPS[2]] = expressed(chain[STEPS[1]])
    chain[STEPS[3]] = chain[STEPS[2]] & tcga_symbols
    return chain


def _protected_genes(directory: UPath) -> set[str]:
    return set(pd.read_parquet(str(directory / "landmark_genes.parquet"))["Symbol"].astype(str)) | CLINICAL_DRIVER_GENES


def create_gene_list(
    data_path: UPath,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    *,
    noise_dataset: str | None = None,
    output: UPath | None = None,
) -> set[str]:
    """Build the gene list and write it to ``<output>/tcga_autoencoder_genes_available.parquet``.

    :param data_path: Directory with one ``<DATASET>.h5mu`` per dataset.
    :param datasets: Datasets whose expression views the list must be measured in.
    :param noise_dataset: Dataset whose expression defines noise; defaults to the first of *datasets*.
    :param output: Directory to write to; defaults to the directory of this script.
    :returns: The genes of the list.
    :raises InvalidInputError: If the noise dataset is not in ``log2(TPM + 1)``.
    """
    directory = UPath(__file__).resolve().parent
    noise_dataset = noise_dataset or datasets[0]
    read = {name: read_expression(data_path / f"{name}.h5mu", with_values=name == noise_dataset) for name in datasets}
    genes_per_dataset = [set(genes) for genes, _, _ in read.values()]
    noise_genes, noise_values, unit = read[noise_dataset]
    if unit != NOISE_UNIT or noise_values is None:
        msg = f"Noise dataset {noise_dataset!r} must have expression_unit {NOISE_UNIT!r}, got {unit!r}."
        raise InvalidInputError(msg)
    protected = _protected_genes(directory)

    def expressed(genes: set[str]) -> set[str]:
        judged = [gene for gene in noise_genes if gene in genes]
        return noise_filter(noise_values[judged], protected) | (genes - set(judged))

    chain = build_chain(genes_per_dataset, download_manifest_genes(), expressed, download_probemap_genes())
    previous: set[str] = set()
    for step, genes in chain.items():
        print(f"{step}: {len(genes)} genes" + (f" (dropped {len(previous - genes)})" if previous else ""))
        previous = genes
    target = (output or directory) / f"{GENE_LIST_NAME}.parquet"
    pd.DataFrame({"Symbol": sorted(previous)}).to_parquet(str(target), index=False, compression="zstd")
    return previous


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and write the gene list.

    :param argv: Argument list, defaulting to ``sys.argv[1:]``.
    """
    from drevalpy.data._paths import get_default_data_dir

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", default=None, help="Directory with <DATASET>.h5mu (default: drevalpy cache).")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="Datasets the list covers.")
    parser.add_argument("--noise-dataset", default=None, help="Dataset defining noise (default: the first dataset).")
    parser.add_argument("--output-dir", default=None, help="Where to write the CSV (default: next to this script).")
    args = parser.parse_args(argv)
    create_gene_list(
        UPath(args.data_path) if args.data_path else get_default_data_dir(),
        tuple(args.datasets),
        noise_dataset=args.noise_dataset,
        output=UPath(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
