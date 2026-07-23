"""SparseGO input file featurizer.

Generates the two files required by SparseGOModel that are not part of the
standard drevalpy dataset:

- gene2ind.txt: tab-delimited index and gene_symbol
- sparseGO_ont.txt: tab-delimited parent, child, and type

Requirements: pip install mygene obonet networkx
"""

from __future__ import annotations

import argparse
import itertools
import math
import os

import networkx as nx
import networkx.algorithms.components.connected as nxacc
import numpy as np
import pandas as pd


def _remove_node(g: nx.DiGraph, node: str) -> nx.DiGraph:
    """Remove a node and reconnect its parents directly to its children.

    This keeps the graph connected: grandparent -> grandchild edges are added
    before the node is deleted, so no paths are broken.

    :param g: directed graph (parent -> child direction after .reverse())
    :param node: GO term ID to remove
    :return: modified graph (same object)
    """
    parents = [src for src, _ in g.in_edges(node)]
    children = [dst for _, dst in g.out_edges(node)]
    new_edges = [(src, dst) for src, dst in itertools.product(parents, children) if src != dst]
    g.add_edges_from(new_edges)
    g.remove_node(node)
    return g


def _build_level_list(g: nx.DiGraph) -> list[list[str]]:
    """Iteratively peel leaves to get nodes organised by level.

    Level 0 = leaves (genes or bottom-most GO terms with no children).
    Each successive level contains the new leaves after removing the previous.

    :param g: directed graph
    :return: list of node lists, one per level bottom-up
    """
    g_copy = g.copy()
    level_list: list[list[str]] = []
    while True:
        leaves = [n for n in g_copy.nodes() if g_copy.out_degree(n) == 0]
        if not leaves:
            break
        level_list.append(leaves)
        g_copy.remove_nodes_from(leaves)
    return level_list


def _download_obo(url: str, dest: str) -> None:
    """Download a file with a browser-like User-Agent to avoid 403 errors.

    :param url: URL to download
    :param dest: local destination path
    :raises RuntimeError: if requests is not installed and download fails
    """
    try:
        import requests

        headers = {"User-Agent": "Mozilla/5.0 (compatible; drevalpy-sparsego/1.0)"}
        response = requests.get(url, headers=headers, timeout=120, stream=True)
        response.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                fh.write(chunk)
        print(f"  Saved to {dest} ({os.path.getsize(dest) // 1024} KB)")
    except ImportError as exc:
        raise RuntimeError("requests is required to download go-basic.obo: pip install requests") from exc


def _load_gene_list(data_path: str, dataset_name: str) -> list[str]:
    """Return gene symbols from the columns of gene_expression.csv.

    :param data_path: root data directory
    :param dataset_name: dataset sub-directory
    :return: list of gene symbols in original column order
    :raises FileNotFoundError: if gene_expression.csv is not found
    """
    expr_file = os.path.join(data_path, dataset_name, "gene_expression.csv")
    if not os.path.exists(expr_file):
        raise FileNotFoundError(
            f"gene_expression.csv not found at {expr_file}. Run standard drevalpy data download first."
        )
    genes = pd.read_csv(expr_file, index_col=0, nrows=0).columns.tolist()
    print(f"Found {len(genes)} genes in gene_expression.csv")
    return genes


def _fetch_gene_go_annotations(genes: list[str]) -> pd.DataFrame:
    """Query MyGene.info and return a DataFrame of (go_term, gene_symbol) pairs.

    Uses a two-step approach: symbol -> entrezgene ID, then entrezgene -> GO BP.
    The query is split in two halves to avoid Bad Gateway errors on large sets.

    :param genes: list of gene symbols
    :return: DataFrame with columns [0=go_term_id, 1=gene_symbol], no duplicates
    """
    try:
        import mygene
    except ImportError as exc:
        msg = "mygene is required. Run: pip install 'drevalpy[sparsego]' (or: pip install mygene)"
        raise ImportError(msg) from exc

    mg = mygene.MyGeneInfo()

    print(f"Querying MyGene.info: {len(genes)} symbols -> entrezgene IDs ...")
    genes_ids: pd.DataFrame = mg.querymany(
        genes,
        scopes="symbol",
        species="human",
        fields="entrezgene",
        as_dataframe=True,
    )
    genes_ids.reset_index(level=0, inplace=True)
    genes_ids.dropna(subset=["entrezgene"], inplace=True)
    genes_ids.drop_duplicates(subset=["query"], inplace=True)
    print(f"  Mapped {len(genes_ids)} / {len(genes)} genes to entrezgene IDs")

    total = len(genes_ids["entrezgene"])
    split = math.ceil(total / 2)
    first_half = genes_ids["entrezgene"].iloc[:split]
    second_half = genes_ids["entrezgene"].iloc[split:]

    print("Querying MyGene.info: entrezgene -> GO BP annotations (2 batches) ...")
    ann1: pd.DataFrame = mg.getgenes(first_half, fields="symbol,go.BP.id", as_dataframe=True)
    ann2: pd.DataFrame = mg.getgenes(second_half, fields="symbol,go.BP.id", as_dataframe=True)
    genes_annotations = pd.concat([ann1, ann2])
    genes_annotations["symbol_ori"] = genes_ids["query"].values

    gene_go: list[tuple[str, str]] = []
    for _, row in genes_annotations.iterrows():
        annotations = row.get("go.BP")
        symbol = row["symbol_ori"]

        if isinstance(annotations, list):
            terms = [item for sublist in annotations for item in sublist.values()]
            pairs = [
                (term, symbol) for term in terms if term != symbol and isinstance(term, str) and term.startswith("GO:")
            ]
            gene_go.extend(pairs)
        elif isinstance(row.get("go.BP.id"), str):
            gene_go.append((row["go.BP.id"], symbol))

    gene_go_df = pd.DataFrame(gene_go).drop_duplicates()
    print(f"  Gene-GO pairs collected: {len(gene_go_df)}")
    return gene_go_df


def _build_pruned_graph(
    gene_go_df: pd.DataFrame,
    obo_file: str | None,
    n: int,
    m: int,
    p: int,
) -> nx.DiGraph:
    """Build the GO hierarchy and prune it according to conditions n, m, p.

    Mirrors the original get_gene_hierarchy.py logic exactly.

    :param gene_go_df: DataFrame with columns [0=go_term, 1=gene_symbol]
    :param obo_file: path to go-basic.obo (downloaded if None)
    :param n: minimum directly-annotated genes per term
    :param m: minimum extra genes a parent must have over each child
    :param p: max levels above the bottom layer to keep
    :return: pruned directed graph (parent -> child, genes as leaf nodes)
    """
    try:
        import obonet
    except ImportError as exc:
        msg = "obonet is required. Run: pip install 'drevalpy[sparsego]' (or: pip install obonet)"
        raise ImportError(msg) from exc

    if obo_file is None:
        obo_url = "https://current.geneontology.org/ontology/go-basic.obo"
        obo_file = "go-basic.obo"
        if not os.path.exists(obo_file):
            print(f"Downloading go-basic.obo from {obo_url} ...")
            _download_obo(obo_url, obo_file)
        else:
            print(f"Using cached {obo_file}")

    print(f"Parsing {obo_file} ...")
    full_graph: nx.MultiDiGraph = obonet.read_obo(obo_file)
    full_graph = full_graph.reverse()
    roots = [n_id for n_id in full_graph.nodes if full_graph.in_degree(n_id) == 0]
    print(f"OBO graph loaded: {len(full_graph.nodes)} nodes, roots: {roots[:5]}")

    all_nodes: set[str] = set(full_graph.nodes())
    keep_nodes: set[str] = set(gene_go_df.iloc[:, 0])

    for term in keep_nodes.copy():
        if full_graph.in_degree(term) == 0:
            keep_nodes.discard(term)

    keep_nodes.add("GO:0008150")

    unwanted = all_nodes - keep_nodes
    our_graph: nx.DiGraph = full_graph.copy()
    print(f"Removing {len(unwanted)} non-annotated terms ...")
    for term in unwanted:
        if term in our_graph:
            _remove_node(our_graph, term)

    gene_go_list = list(gene_go_df.itertuples(index=False, name=None))
    our_graph.add_edges_from(gene_go_list)

    for node in list(our_graph.nodes):
        if our_graph.in_degree(node) == 0 and node != "GO:0008150":
            _remove_node(our_graph, node)

    roots_now = [n_id for n_id in our_graph.nodes if our_graph.in_degree(n_id) == 0]
    print(f"After adding genes: {len(our_graph.nodes)} nodes, roots: {roots_now[:3]}")

    level_list = _build_level_list(our_graph)
    print(f"Graph depth: {len(level_list)} levels. Applying conditions n={n}, m={m} ...")

    for terms_to_check in level_list[1:]:
        for node in list(terms_to_check):
            if node not in our_graph:
                continue

            genes: list[str] = []
            children: list[str] = []
            for _, child in our_graph.out_edges(node):
                if child.startswith("GO:"):
                    children.append(child)
                else:
                    genes.append(child)
            genes = list(set(genes))
            children = list(set(children))

            if len(genes) < n and node != "GO:0008150":
                _remove_node(our_graph, node)
                continue

            if children:
                for child_node in children:
                    child_genes = [c for _, c in our_graph.out_edges(child_node) if not c.startswith("GO:")]
                    if len(set(genes) - set(child_genes)) < m and node != "GO:0008150":
                        _remove_node(our_graph, node)
                        break

    roots_after_nm = [n_id for n_id in our_graph.nodes if our_graph.in_degree(n_id) == 0]
    print(f"After n/m pruning: {len(our_graph.nodes)} nodes, roots: {roots_after_nm[:3]}")

    level_list_pruned = _build_level_list(our_graph)
    print(f"Depth after n/m: {len(level_list_pruned)} levels. Applying p={p} ...")
    for level in level_list_pruned[p + 1 : len(level_list_pruned) - 1]:  # noqa: E203
        for term in level:
            if term in our_graph:
                _remove_node(our_graph, term)

    uG = our_graph.to_undirected()
    components = list(nxacc.connected_components(uG))
    final_roots = [n_id for n_id in our_graph.nodes if our_graph.in_degree(n_id) == 0]
    print(f"Final graph: {len(our_graph.nodes)} nodes, {len(components)} component(s), roots: {final_roots[:3]}")
    if len(components) > 1:
        print("WARNING: more than one connected component. load_ontology will fail. Consider adjusting n/m/p.")

    return our_graph


def _write_outputs(our_graph: nx.DiGraph, data_path: str, dataset_name: str) -> None:
    """Write gene2ind.txt and sparseGO_ont.txt.

    :param our_graph: final pruned graph (parent -> child edges)
    :param data_path: root data directory
    :param dataset_name: dataset sub-directory
    """
    out_dir = os.path.join(data_path, dataset_name)

    edges = np.array(list(our_graph.edges()))
    edges = np.unique(edges, axis=0)
    type_col = np.where(
        np.char.startswith(edges[:, 1].astype(str), "GO:"),
        "default",
        "gene",
    )
    edges_with_type = np.column_stack([edges, type_col])

    ont_path = os.path.join(out_dir, "sparseGO_ont.txt")
    pd.DataFrame(edges_with_type).to_csv(ont_path, sep="\t", index=False, header=False)
    n_default = (type_col == "default").sum()
    n_gene = (type_col == "gene").sum()
    print(f"Wrote {n_default} term-term + {n_gene} gene-term edges -> {ont_path}")

    gene_edges = edges_with_type[edges_with_type[:, 2] == "gene"]
    keep_genes = sorted(set(gene_edges[:, 1]))
    print(f"Genes in ontology: {len(keep_genes)}")

    gene2id = {gene: idx for idx, gene in enumerate(keep_genes)}
    gene2ind_path = os.path.join(out_dir, "gene2ind.txt")
    with open(gene2ind_path, "w") as fh:
        for gene, idx in gene2id.items():
            fh.write(f"{idx}\t{gene}\n")
    print(f"Wrote {len(gene2id)} genes -> {gene2ind_path}")

    expr_path = os.path.join(out_dir, "gene_expression.csv")
    expr_cols = set(pd.read_csv(expr_path, index_col=0, nrows=0).columns.tolist())
    missing = [g for g in keep_genes if g not in expr_cols]
    if missing:
        print(f"  WARNING: {len(missing)} ontology genes missing from gene_expression.csv: {missing}")
    else:
        print(f"  OK: all {len(keep_genes)} ontology genes present in gene_expression.csv")


def create_sparsego_files(
    data_path: str,
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
    out_dir = os.path.join(data_path, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    genes = _load_gene_list(data_path, dataset_name)
    gene_go_df = _fetch_gene_go_annotations(genes)
    our_graph = _build_pruned_graph(gene_go_df, obo_file, n=n, m=m, p=p)
    _write_outputs(our_graph, data_path, dataset_name)

    print("\nDone.")
    print(f"  {os.path.join(out_dir, 'gene2ind.txt')}")
    print(f"  {os.path.join(out_dir, 'sparseGO_ont.txt')}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate gene2ind.txt and sparseGO_ont.txt for a drevalpy dataset.")
    parser.add_argument("dataset_name", help="Dataset name, e.g. CTRPv2")
    parser.add_argument("--data_path", default="data", help="Root data directory")
    parser.add_argument("--obo_file", default=None, help="Path to go-basic.obo (auto-downloaded if not provided)")
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
