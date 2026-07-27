"""GO graph construction and MyGene annotation for SparseGO featurizer files."""

from __future__ import annotations

import itertools
import math
import os

import networkx as nx
import networkx.algorithms.components.connected as nxacc
import pandas as pd

GO_ROOT = "GO:0008150"


def _remove_node(g: nx.DiGraph, node: str) -> nx.DiGraph:
    """Remove a node and reconnect its parents directly to its children."""
    parents = [src for src, _ in g.in_edges(node)]
    children = [dst for _, dst in g.out_edges(node)]
    new_edges = [(src, dst) for src, dst in itertools.product(parents, children) if src != dst]
    g.add_edges_from(new_edges)
    g.remove_node(node)
    return g


def build_level_list(g: nx.DiGraph) -> list[list[str]]:
    """Iteratively peel leaves to get nodes organised by level."""
    g_copy = g.copy()
    level_list: list[list[str]] = []
    while True:
        leaves = [n for n in g_copy.nodes() if g_copy.out_degree(n) == 0]
        if not leaves:
            break
        level_list.append(leaves)
        g_copy.remove_nodes_from(leaves)
    return level_list


def download_obo(url: str, dest: str) -> None:
    """Download a file with a browser-like User-Agent to avoid 403 errors."""
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


def _pairs_from_go_bp_row(row: pd.Series) -> list[tuple[str, str]]:
    symbol = row["symbol_ori"]
    annotations = row.get("go.BP")
    if isinstance(annotations, list):
        terms = [item for sublist in annotations for item in sublist.values()]
        return [(term, symbol) for term in terms if term != symbol and isinstance(term, str) and term.startswith("GO:")]
    if isinstance(row.get("go.BP.id"), str):
        return [(row["go.BP.id"], symbol)]
    return []


def fetch_gene_go_annotations(genes: list[str]) -> pd.DataFrame:
    """Query MyGene.info and return a DataFrame of (go_term, gene_symbol) pairs."""
    try:
        import mygene
    except ImportError as exc:
        msg = "mygene is required. Reinstall drevalpy (mygene is a core dependency), or: pip install mygene"
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
        gene_go.extend(_pairs_from_go_bp_row(row))

    gene_go_df = pd.DataFrame(gene_go).drop_duplicates()
    print(f"  Gene-GO pairs collected: {len(gene_go_df)}")
    return gene_go_df


def _ensure_obo_path(obo_file: str | None) -> str:
    if obo_file is not None:
        return obo_file
    obo_url = "https://current.geneontology.org/ontology/go-basic.obo"
    obo_file = "go-basic.obo"
    if not os.path.exists(obo_file):
        print(f"Downloading go-basic.obo from {obo_url} ...")
        download_obo(obo_url, obo_file)
    else:
        print(f"Using cached {obo_file}")
    return obo_file


def _load_reversed_obo(obo_file: str) -> nx.MultiDiGraph:
    try:
        import obonet
    except ImportError as exc:
        msg = "obonet is required. Reinstall drevalpy (obonet is a core dependency), or: pip install obonet"
        raise ImportError(msg) from exc

    print(f"Parsing {obo_file} ...")
    full_graph: nx.MultiDiGraph = obonet.read_obo(obo_file)
    full_graph = full_graph.reverse()
    roots = [n_id for n_id in full_graph.nodes if full_graph.in_degree(n_id) == 0]
    print(f"OBO graph loaded: {len(full_graph.nodes)} nodes, roots: {roots[:5]}")
    return full_graph


def _annotated_keep_nodes(full_graph: nx.MultiDiGraph, gene_go_df: pd.DataFrame) -> set[str]:
    keep_nodes: set[str] = set(gene_go_df.iloc[:, 0])
    for term in keep_nodes.copy():
        if full_graph.in_degree(term) == 0:
            keep_nodes.discard(term)
    keep_nodes.add(GO_ROOT)
    return keep_nodes


def _prune_unannotated_terms(full_graph: nx.MultiDiGraph, keep_nodes: set[str]) -> nx.DiGraph:
    unwanted = set(full_graph.nodes()) - keep_nodes
    our_graph: nx.DiGraph = full_graph.copy()
    print(f"Removing {len(unwanted)} non-annotated terms ...")
    for term in unwanted:
        if term in our_graph:
            _remove_node(our_graph, term)
    return our_graph


def _attach_genes_and_drop_orphan_roots(our_graph: nx.DiGraph, gene_go_df: pd.DataFrame) -> None:
    gene_go_list = list(gene_go_df.itertuples(index=False, name=None))
    our_graph.add_edges_from(gene_go_list)
    for node in list(our_graph.nodes):
        if our_graph.in_degree(node) == 0 and node != GO_ROOT:
            _remove_node(our_graph, node)


def _direct_genes_and_children(graph: nx.DiGraph, node: str) -> tuple[list[str], list[str]]:
    genes: list[str] = []
    children: list[str] = []
    for _, child in graph.out_edges(node):
        if child.startswith("GO:"):
            children.append(child)
        else:
            genes.append(child)
    return list(set(genes)), list(set(children))


def _child_gene_set(graph: nx.DiGraph, child_node: str) -> set[str]:
    child_genes = [c for _, c in graph.out_edges(child_node) if not c.startswith("GO:")]
    return set(child_genes)


def _should_remove_nm_node(
    graph: nx.DiGraph,
    node: str,
    genes: list[str],
    children: list[str],
    *,
    n: int,
    m: int,
) -> bool:
    if len(genes) < n and node != GO_ROOT:
        return True
    for child_node in children:
        child_genes = _child_gene_set(graph, child_node)
        if len(set(genes) - child_genes) < m and node != GO_ROOT:
            return True
    return False


def _apply_nm_pruning(graph: nx.DiGraph, level_list: list[list[str]], *, n: int, m: int) -> None:
    print(f"Graph depth: {len(level_list)} levels. Applying conditions n={n}, m={m} ...")
    for terms_to_check in level_list[1:]:
        for node in list(terms_to_check):
            if node not in graph:
                continue
            genes, children = _direct_genes_and_children(graph, node)
            if _should_remove_nm_node(graph, node, genes, children, n=n, m=m):
                _remove_node(graph, node)


def _apply_p_pruning(graph: nx.DiGraph, *, p: int) -> None:
    level_list_pruned = build_level_list(graph)
    print(f"Depth after n/m: {len(level_list_pruned)} levels. Applying p={p} ...")
    for level in level_list_pruned[p + 1 : len(level_list_pruned) - 1]:  # noqa: E203
        for term in level:
            if term in graph:
                _remove_node(graph, term)


def _report_connectivity(graph: nx.DiGraph) -> None:
    uG = graph.to_undirected()
    components = list(nxacc.connected_components(uG))
    final_roots = [n_id for n_id in graph.nodes if graph.in_degree(n_id) == 0]
    print(f"Final graph: {len(graph.nodes)} nodes, {len(components)} component(s), roots: {final_roots[:3]}")
    if len(components) > 1:
        print("WARNING: more than one connected component. load_ontology will fail. Consider adjusting n/m/p.")


def build_pruned_graph(
    gene_go_df: pd.DataFrame,
    obo_file: str | None,
    n: int,
    m: int,
    p: int,
) -> nx.DiGraph:
    """Build the GO hierarchy and prune it according to conditions n, m, p."""
    obo_path = _ensure_obo_path(obo_file)
    full_graph = _load_reversed_obo(obo_path)
    keep_nodes = _annotated_keep_nodes(full_graph, gene_go_df)
    our_graph = _prune_unannotated_terms(full_graph, keep_nodes)
    _attach_genes_and_drop_orphan_roots(our_graph, gene_go_df)

    roots_now = [n_id for n_id in our_graph.nodes if our_graph.in_degree(n_id) == 0]
    print(f"After adding genes: {len(our_graph.nodes)} nodes, roots: {roots_now[:3]}")

    level_list = build_level_list(our_graph)
    _apply_nm_pruning(our_graph, level_list, n=n, m=m)

    roots_after_nm = [n_id for n_id in our_graph.nodes if our_graph.in_degree(n_id) == 0]
    print(f"After n/m pruning: {len(our_graph.nodes)} nodes, roots: {roots_after_nm[:3]}")

    _apply_p_pruning(our_graph, p=p)
    _report_connectivity(our_graph)
    return our_graph
