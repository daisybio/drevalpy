"""SparseGO input file featurizer.

Generates SparseGO ontology data from gene expression in a .h5mu file
and writes the result to mdata.uns["sparsego"].

Requirements: pip install mygene obonet networkx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
from sparsego_graph import build_pruned_graph, fetch_gene_go_annotations

import mudata as md


def _load_gene_list_from_mudata(mdata: md.MuData) -> list[str]:
    """Return gene symbols from the gene_expression modality."""
    if "gene_expression" not in mdata.mod:
        msg = "MuData must contain a 'gene_expression' modality."
        raise ValueError(msg)
    return list(mdata.mod["gene_expression"].var_names)


def _build_ontology_arrays(our_graph: nx.DiGraph) -> dict[str, object]:
    """Convert the pruned graph into serializable arrays for mdata.uns."""
    edges = np.array(list(our_graph.edges()))
    edges = np.unique(edges, axis=0)
    type_col = np.where(
        np.char.startswith(edges[:, 1].astype(str), "GO:"),
        "default",
        "gene",
    )

    gene_edges = edges[type_col == "gene"]
    keep_genes = sorted(set(gene_edges[:, 1]))
    gene2ind = {gene: idx for idx, gene in enumerate(keep_genes)}

    return {
        "edges_parent": edges[:, 0].tolist(),
        "edges_child": edges[:, 1].tolist(),
        "edges_type": type_col.tolist(),
        "gene2ind": gene2ind,
        "genes": keep_genes,
    }


def main(h5mu_path: Path, *, obo_file: str | None = None, n: int = 5, m: int = 10, p: int = 8) -> None:
    """Generate SparseGO ontology data and write to mdata.uns['sparsego'].

    :param h5mu_path: Path to the .h5mu file.
    :param obo_file: Path to go-basic.obo (auto-downloaded if None).
    :param n: Minimum directly-annotated genes per GO term.
    :param m: Minimum extra genes parent must have over each child.
    :param p: Max parent-child levels above bottom layer.
    """
    mdata = md.read(str(h5mu_path))
    genes = _load_gene_list_from_mudata(mdata)

    print(f"Found {len(genes)} genes in gene_expression modality")
    gene_go_df = fetch_gene_go_annotations(genes)
    our_graph = build_pruned_graph(gene_go_df, obo_file, n=n, m=m, p=p)

    ontology_data = _build_ontology_arrays(our_graph)
    mdata.uns["sparsego"] = ontology_data

    mdata.write(str(h5mu_path))
    n_genes = len(ontology_data["genes"])
    n_edges = len(ontology_data["edges_parent"])
    print(f"Wrote SparseGO ontology ({n_genes} genes, {n_edges} edges) to mdata.uns['sparsego'] in {h5mu_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SparseGO ontology data and store in .h5mu.")
    parser.add_argument("h5mu_path", type=Path, help="Path to the .h5mu file")
    parser.add_argument("--obo_file", type=Path, default=None, help="Path to go-basic.obo (auto-downloaded if omitted)")
    parser.add_argument("--n", type=int, default=5, help="Min directly-annotated genes per GO term")
    parser.add_argument("--m", type=int, default=10, help="Min extra genes parent must have over each child")
    parser.add_argument("--p", type=int, default=8, help="Max levels above bottom GO layer")
    args = parser.parse_args()
    main(args.h5mu_path, obo_file=str(args.obo_file) if args.obo_file else None, n=args.n, m=args.m, p=args.p)
