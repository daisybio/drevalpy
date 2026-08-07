"""SparseGO input file featurizer.

Generates the two files required by SparseGOModel that are not part of the
standard drevalpy dataset:

- gene2ind.txt: tab-delimited index and gene_symbol
- sparseGO_ont.txt: tab-delimited parent, child, and type

Requirements: pip install mygene obonet networkx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .sparsego_graph import build_pruned_graph, fetch_gene_go_annotations


def _load_gene_list(data_path: Path, dataset_name: str) -> list[str]:
    """Return gene symbols from the columns of gene_expression.csv.

    :param data_path: root data directory
    :param dataset_name: dataset sub-directory
    :return: list of gene symbols in original column order
    :raises FileNotFoundError: if gene_expression.csv is not found
    """
    expr_file = data_path / dataset_name / "gene_expression.csv"
    if not expr_file.exists():
        raise FileNotFoundError(
            f"gene_expression.csv not found at {expr_file}. Run standard drevalpy data download first."
        )
    genes = pd.read_csv(expr_file, index_col=0, nrows=0).columns.tolist()
    print(f"Found {len(genes)} genes in gene_expression.csv")
    return genes


def _write_outputs(our_graph: nx.DiGraph, data_path: Path, dataset_name: str) -> None:
    """Write gene2ind.txt and sparseGO_ont.txt.

    :param our_graph: final pruned graph (parent -> child edges)
    :param data_path: root data directory
    :param dataset_name: dataset sub-directory
    """
    out_dir = data_path / dataset_name

    edges = np.array(list(our_graph.edges()))
    edges = np.unique(edges, axis=0)
    type_col = np.where(
        np.char.startswith(edges[:, 1].astype(str), "GO:"),
        "default",
        "gene",
    )
    edges_with_type = np.column_stack([edges, type_col])

    ont_path = out_dir / "sparseGO_ont.txt"
    pd.DataFrame(edges_with_type).to_csv(ont_path, sep="\t", index=False, header=False)
    n_default = (type_col == "default").sum()
    n_gene = (type_col == "gene").sum()
    print(f"Wrote {n_default} term-term + {n_gene} gene-term edges -> {ont_path}")

    gene_edges = edges_with_type[edges_with_type[:, 2] == "gene"]
    keep_genes = sorted(set(gene_edges[:, 1]))
    print(f"Genes in ontology: {len(keep_genes)}")

    gene2id = {gene: idx for idx, gene in enumerate(keep_genes)}
    gene2ind_path = out_dir / "gene2ind.txt"
    with open(gene2ind_path, "w") as fh:
        for gene, idx in gene2id.items():
            fh.write(f"{idx}\t{gene}\n")
    print(f"Wrote {len(gene2id)} genes -> {gene2ind_path}")

    expr_path = out_dir / "gene_expression.csv"
    expr_cols = set(pd.read_csv(expr_path, index_col=0, nrows=0).columns.tolist())
    missing = [g for g in keep_genes if g not in expr_cols]
    if missing:
        print(f"  WARNING: {len(missing)} ontology genes missing from gene_expression.csv: {missing}")
    else:
        print(f"  OK: all {len(keep_genes)} ontology genes present in gene_expression.csv")


def create_sparsego_files(
    data_path: str | Path,
    dataset_name: str,
    obo_file: str | None = None,
    n: int = 5,
    m: int = 10,
    p: int = 8,
) -> None:
    """Generate gene2ind.txt and sparseGO_ont.txt for a drevalpy dataset.

    Mirrors the logic of the original SparseGO get_gene_hierarchy.py exactly,
    including the three pruning conditions n, m, p and the two-step MyGene
    query via entrezgene IDs.

    Does NOT modify gene_expression.csv: drevalpy selects genes by column
    name at runtime via get_feature_matrix.

    :param data_path: root data directory
    :param dataset_name: dataset name (sub-directory under data_path)
    :param obo_file: path to go-basic.obo (auto-downloaded if None)
    :param n: minimum directly-annotated genes per GO term (default 5)
    :param m: minimum extra genes parent must have over each child (default 10)
    :param p: max parent-child levels above bottom layer (default 8)
    """
    data_root = Path(data_path)
    out_dir = data_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    genes = _load_gene_list(data_root, dataset_name)
    gene_go_df = fetch_gene_go_annotations(genes)
    our_graph = build_pruned_graph(gene_go_df, obo_file, n=n, m=m, p=p)
    _write_outputs(our_graph, data_root, dataset_name)

    print("\nDone.")
    print(f"  {out_dir / 'gene2ind.txt'}")
    print(f"  {out_dir / 'sparseGO_ont.txt'}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate gene2ind.txt and sparseGO_ont.txt for a drevalpy dataset.")
    parser.add_argument("dataset_name", help="Dataset name, e.g. CTRPv2")
    parser.add_argument("--data_path", type=Path, default=Path("data"), help="Root data directory")
    parser.add_argument(
        "--obo_file", type=Path, default=None, help="Path to go-basic.obo (auto-downloaded if not provided)"
    )
    parser.add_argument("--n", type=int, default=5, help="Min directly-annotated genes per GO term (default: 5)")
    parser.add_argument(
        "--m", type=int, default=10, help="Min extra genes parent must have over each child (default: 10)"
    )
    parser.add_argument("--p", type=int, default=8, help="Max levels above bottom GO layer (default: 8)")
    args = parser.parse_args()

    create_sparsego_files(
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        obo_file=args.obo_file,
        n=args.n,
        m=args.m,
        p=args.p,
    )


if __name__ == "__main__":
    main()
