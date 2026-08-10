r"""GSVA pathway-score featurizer.

Computes GSVA pathway-activity scores from gene expression in a .h5mu file
and writes them to response.obsm["pathway_features"].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mudata as md
import numpy as np
import pandas as pd


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
    """Run gseapy GSVA and return a [samples x pathways] DataFrame."""
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
    long = gv.res2d.copy()
    cols = {c.lower(): c for c in long.columns}
    term_col = cols.get("term", "Term")
    name_col = cols.get("name", "Name")
    es_col = cols.get("es", cols.get("nes", "ES"))
    wide = long.pivot(index=term_col, columns=name_col, values=es_col)
    return wide.T.astype(np.float32)


def main(
    h5mu_path: Path,
    *,
    gene_sets: str,
    min_size: int = 5,
    max_size: int = 2000,
    kcdf: str = "Gaussian",
    mx_diff: bool = True,
    threads: int = 4,
    seed: int = 42,
) -> None:
    """Compute GSVA pathway scores and write to response.obsm['pathway_features'].

    :param h5mu_path: Path to the .h5mu file.
    :param gene_sets: Path to MSigDB .gmt file.
    :param min_size: Minimum gene-set size.
    :param max_size: Maximum gene-set size.
    :param kcdf: Kernel for the CDF.
    :param mx_diff: GSVA mx_diff option.
    :param threads: Parallelism.
    :param seed: Random seed.
    """
    mdata = md.read(str(h5mu_path))

    if "gene_expression" not in mdata.mod:
        msg = "MuData must contain a 'gene_expression' modality."
        raise ValueError(msg)

    ge = mdata.mod["gene_expression"]
    expr = pd.DataFrame(
        ge.X if not hasattr(ge.X, "toarray") else ge.X.toarray(), index=ge.obs_names, columns=ge.var_names
    )

    before = len(expr)
    expr = expr.loc[~expr.index.duplicated(keep="first")]
    if before != len(expr):
        print(f"Removed {before - len(expr)} duplicated cell lines")

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
    )

    response = mdata.mod["response"]

    aligned = np.zeros((len(response.obs_names), scores.shape[1]), dtype=np.float32)
    for i, cl in enumerate(response.obs_names):
        if cl in scores.index:
            aligned[i] = scores.loc[cl].values

    response.obsm["pathway_features"] = aligned
    mdata.write(str(h5mu_path))
    print(f"Wrote pathway features ({aligned.shape}) to response.obsm['pathway_features'] in {h5mu_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute GSVA pathway features and store in .h5mu.")
    parser.add_argument("h5mu_path", type=Path, help="Path to the .h5mu file")
    parser.add_argument("--gene_sets", required=True, help="Path to MSigDB C2 CP .gmt file")
    parser.add_argument("--min_size", type=int, default=5)
    parser.add_argument("--max_size", type=int, default=2000)
    parser.add_argument("--kcdf", default="Gaussian")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        args.h5mu_path,
        gene_sets=args.gene_sets,
        min_size=args.min_size,
        max_size=args.max_size,
        kcdf=args.kcdf,
        seed=args.seed,
    )
