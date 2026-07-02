r"""
GSVA pathway-score featurizer for Precily.

Computes GSVA pathway-activity scores for cell lines of a dataset in a
single pass and writes them to a CSV

Input : data/<dataset>/gene_expression.csv   (cell lines x genes)
Output: data/<dataset>/precily_pathways.csv   (cell lines x pathways)

    python -m drevalpy.datasets.featurizer.create_precily_pathway_features GDSC2 \\
        --gene_sets data/msigdb/c2.cp.v6.1.symbols.gmt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def _load_gene_expression(data_path: str, dataset_name: str) -> pd.DataFrame:
    """
    Load the dataset's gene-expression matrix.

    :param data_path: root data path
    :param dataset_name: dataset name
    :return: DataFrame, cell lines in rows, genes in columns
    :raises FileNotFoundError: if the expression file is not found
    """
    expr_file = os.path.join(data_path, dataset_name, "gene_expression.csv")
    if not os.path.exists(expr_file):
        raise FileNotFoundError(f"{expr_file} not found.")
    df = pd.read_csv(expr_file, index_col=0)
    df = df.select_dtypes(include="number")
    return df


def _run_gsva(
    expr_genes_by_samples: pd.DataFrame,
    gene_sets: str,
    min_size: int,
    max_size: int,
    kcdf: str,
    mx_diff: bool,
    threads: int,
    seed: int,
) -> pd.DataFrame:
    """
    Run gseapy GSVA and return a [samples x pathways] DataFrame.

    :param expr_genes_by_samples: genes in rows, samples in columns
    :param gene_sets: path to .gmt (MSigDB C2 CP v6.1)
    :param min_size: minimum gene-set size
    :param max_size: maximum gene-set size
    :param kcdf: "Gaussian" for log2(TPM+1)
    :param mx_diff: GSVA mx_diff option
    :param threads: parallelism
    :param seed: random seed
    :return: cell lines in rows, pathways in columns
    """
    import gseapy as gp

    gv = gp.gsva(
        data=expr_genes_by_samples,
        gene_sets=gene_sets,
        kcdf=kcdf,
        min_size=min_size,
        max_size=max_size,
        mx_diff=mx_diff,
        threads=threads,
        seed=seed,
        outdir=None,
        verbose=False,
    )
    # gseapy returns long format; column names vary across versions.
    long = gv.res2d.copy()
    cols = {c.lower(): c for c in long.columns}
    term_col = cols.get("term", "Term")
    name_col = cols.get("name", "Name")
    es_col = cols.get("es", cols.get("nes", "ES"))

    wide = long.pivot(index=term_col, columns=name_col, values=es_col)  # [pathways x samples]
    return wide.T.astype(np.float32)  # [samples x pathways]


def create_precily_pathway_features(
    data_path: str,
    dataset_name: str,
    gene_sets: str,
    min_size: int = 5,
    max_size: int = 2000,
    kcdf: str = "Gaussian",
    mx_diff: bool = True,
    threads: int = 4,
    seed: int = 42,
) -> None:
    """
    Compute GSVA pathway scores for all cell lines and write precily_pathways.csv.

    :param data_path: root data path
    :param dataset_name: dataset name
    :param gene_sets: path to MSigDB C2 CP v6.1 .gmt file
    :param min_size: minimum gene-set size
    :param max_size: maximum gene-set size
    :param kcdf: kernel for the CDF ("Gaussian" for log2(TPM+1))
    :param mx_diff: GSVA mx_diff option
    :param threads: parallelism
    :param seed: random seed
    """
    expr = _load_gene_expression(data_path, dataset_name)  # [cell lines x genes]

    before = len(expr)
    expr = expr.loc[~expr.index.duplicated(keep="first")]
    print(f"Removed {before - len(expr)} duplicated cell lines")

    # gseapy expects genes in rows, samples in columns.
    expr_genes_by_samples = expr.T

    scores = _run_gsva(
        expr_genes_by_samples,
        gene_sets=gene_sets,
        min_size=min_size,
        max_size=max_size,
        kcdf=kcdf,
        mx_diff=mx_diff,
        threads=threads,
        seed=seed,
    )  # [cell lines x pathways]

    names_file = os.path.join(data_path, dataset_name, "cell_line_names.csv")

    if os.path.exists(names_file):
        names_df = pd.read_csv(names_file)

        if {"cellosaurus_id", "cell_line_name"}.issubset(names_df.columns):
            names = names_df.set_index("cellosaurus_id")["cell_line_name"]
            scores = scores.join(names, how="inner")
            scores = scores.set_index("cell_line_name")
    scores.index.name = "cell_line_name"
    out_path = os.path.join(data_path, dataset_name, "pathway_features.csv")
    scores.to_csv(out_path)
    print(f"Wrote {scores.shape[0]} cell lines x {scores.shape[1]} pathways -> {out_path}")


def main() -> None:
    """Compute and save GSVA pathway features for a dataset."""
    parser = argparse.ArgumentParser(description="GSVA pathway featurizer for Precily.")
    parser.add_argument("dataset_name")
    parser.add_argument("--data_path", default="data")
    parser.add_argument(
        "--gene_sets",
        required=True,
        help="path to MSigDB C2 CP v6.1 .gmt file (c2.cp.v6.1.symbols.gmt)",
    )
    parser.add_argument("--min_size", type=int, default=5)
    parser.add_argument("--max_size", type=int, default=2000)
    parser.add_argument("--kcdf", default="Gaussian")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_precily_pathway_features(
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        gene_sets=args.gene_sets,
        min_size=args.min_size,
        max_size=args.max_size,
        kcdf=args.kcdf,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
