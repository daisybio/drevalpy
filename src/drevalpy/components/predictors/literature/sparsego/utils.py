"""Utility functions for the SparseGO model.

Adapted from https://github.com/KatynaSada/SparseGO_lightning
"""

import logging
from io import StringIO
from typing import Any

import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import numpy as np
import pandas as pd
from upath import UPath as Path

logger = logging.getLogger(__name__)


def load_mapping(mapping_file: str | Path | StringIO) -> dict:
    """Open a txt file with two columns and return a dictionary.

    The second column becomes the key and the first column becomes the value.
    Used to read gene2ind.txt and drug2ind.txt files.

    :param mapping_file: Path to the mapping txt file.

    :returns: Dictionary mapping names (e.g. gene symbols) to integer indices.
    """
    mapping = {}
    for line in _read_lines(mapping_file):
        line_parts = line.rstrip().split()
        mapping[line_parts[1]] = int(line_parts[0])
    return mapping


def _read_ontology_edges(
    ontology_file: str | Path | StringIO, gene2id_mapping: dict
) -> tuple[nx.DiGraph, list[list[str]], list[list[str]], set[str], dict[str, set[int]]]:
    directed_graph = nx.DiGraph()
    terms_pairs: list[list[str]] = []
    genes_terms_pairs: list[list[str]] = []
    gene_set: set[str] = set()
    term_direct_gene_map: dict[str, set[int]] = {}

    for line in _read_lines(ontology_file):
        line_parts = line.rstrip().split()

        if line_parts[2] == "default":
            directed_graph.add_edge(line_parts[0], line_parts[1])
            terms_pairs.append([line_parts[0], line_parts[1]])
            continue

        if line_parts[1] not in gene2id_mapping:
            continue
        genes_terms_pairs.append([line_parts[0], line_parts[1]])
        term_direct_gene_map.setdefault(line_parts[0], set()).add(gene2id_mapping[line_parts[1]])
        gene_set.add(line_parts[1])

    return directed_graph, terms_pairs, genes_terms_pairs, gene_set, term_direct_gene_map


def _annotate_term_gene_counts(directed_graph: nx.DiGraph, term_direct_gene_map: dict[str, set[int]]) -> dict[str, int]:
    term_size_map: dict[str, int] = {}
    for term in directed_graph.nodes():
        term_gene_set: set[int] = set(term_direct_gene_map.get(term, set()))
        for child in nxadag.descendants(directed_graph, term):
            if child in term_direct_gene_map:
                term_gene_set |= term_direct_gene_map[child]
        if len(term_gene_set) == 0:
            raise ValueError(f"SparseGO ontology contains a term without any annotated gene; delete term: {term}")
        term_size_map[term] = len(term_gene_set)
    return term_size_map


def _validate_ontology_topology(directed_graph: nx.DiGraph) -> None:
    leaves = [n for n in directed_graph.nodes if directed_graph.in_degree(n) == 0]
    undirected_graph = directed_graph.to_undirected()
    connected_subgraphs = list(nxacc.connected_components(undirected_graph))

    if not leaves:
        raise ValueError("SparseGO ontology has no term-term edges and therefore no root term")
    logger.info("Ontology has %d root(s): %s", len(leaves), leaves[0])
    logger.info("Ontology has %d terms", len(directed_graph.nodes()))
    logger.info("Ontology has %d connected components", len(connected_subgraphs))

    if len(leaves) > 1:
        raise ValueError(f"SparseGO ontology has {len(leaves)} roots ({leaves[:5]}); use exactly one root")
    if len(connected_subgraphs) > 1:
        raise ValueError(f"SparseGO ontology has {len(connected_subgraphs)} connected components; connect them")


def load_ontology(ontology_file: str | Path | StringIO, gene2id_mapping: dict) -> tuple:
    """Create the directed graph of GO terms and store connected elements in arrays.

    The ontology file is tab-delimited with three columns:
    - Column 1: parent term or gene
    - Column 2: child term or gene
    - Column 3: 'default' for term-term connections, 'gene' for term-gene annotations

    :param ontology_file: Path to the ontology file (e.g. sparseGO_ont.txt).
    :param gene2id_mapping: Dictionary mapping gene names to integer IDs.

    :returns: Tuple of (directed graph, term-term pairs array, gene-term pairs array).
    :raises ValueError: If a term has no annotated genes, the ontology has no root, several roots, or
        several connected components.
    """
    directed_graph, terms_pairs, genes_terms_pairs, gene_set, term_direct_gene_map = _read_ontology_edges(
        ontology_file, gene2id_mapping
    )

    terms_pairs_arr: np.ndarray = np.array(terms_pairs)
    genes_terms_pairs_arr: np.ndarray = np.array(genes_terms_pairs)

    logger.info("Ontology annotates %d genes", len(gene_set))

    _annotate_term_gene_counts(directed_graph, term_direct_gene_map)
    _validate_ontology_topology(directed_graph)

    return directed_graph, terms_pairs_arr, genes_terms_pairs_arr


def sort_pairs(
    genes_terms_pairs: np.ndarray,
    terms_pairs: np.ndarray,
    directed_graph: nx.DiGraph,
    gene2id_mapping: dict,
) -> tuple:
    """Concatenate and sort all pairs so the parent term is always in the first column.

    :param genes_terms_pairs: Array of (term, gene) pairs.
    :param terms_pairs: Array of (parent_term, child_term) pairs.
    :param directed_graph: Directed graph of GO terms.
    :param gene2id_mapping: Dictionary mapping gene names to integer IDs.

    :returns: Tuple of (sorted_pairs, level_list, level_number).
    """
    all_pairs = np.concatenate((genes_terms_pairs, terms_pairs))
    graph = directed_graph.copy()

    level_list = []
    level_list.append(list(gene2id_mapping.keys()))  # genes are level 0

    while True:
        leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        if len(leaves) == 0:
            break
        level_list.append(leaves)
        graph.remove_nodes_from(leaves)

    level_number: dict[str, int] = {}
    for i, layer in enumerate(level_list):
        for item in layer:
            level_number[item] = i

    sorted_pairs = all_pairs.copy()
    for i, pair in enumerate(sorted_pairs):
        level1 = level_number[pair[0]]
        level2 = level_number[pair[1]]
        if level2 > level1:  # ensure parent term is always first
            sorted_pairs[i][1] = all_pairs[i][0]
            sorted_pairs[i][0] = all_pairs[i][1]

    return sorted_pairs, level_list, level_number


def pairs_in_layers(sorted_pairs: np.ndarray, level_list: list, level_number: dict) -> list:
    """Divide all pairs by layers and add virtual nodes for non-adjacent connections.

    In GO, a term can have parents at non-adjacent levels. Virtual nodes are
    inserted to ensure all connections are between adjacent layers, which is
    required by the SparseLinear layer implementation.

    :param sorted_pairs: Array of (parent, child) pairs sorted by level.
    :param level_list: List of term sets at each level.
    :param level_number: Dictionary mapping each term/gene to its level index.

    :returns: List of numpy arrays, one per layer, each containing (parent, child) pairs.
    """
    total_layers = len(level_list) - 1
    layer_connections: list[list | np.ndarray] = [[] for _ in range(total_layers)]

    for pair in sorted_pairs:
        parent = level_number[pair[0]]
        child = level_number[pair[1]]

        layer_connections[child].append(pair)

        dif = parent - child
        if dif != 1:  # non-adjacent levels: add virtual nodes
            virtual_node_layer = parent - 1
            for _ in range(dif - 1):
                layer_connections[virtual_node_layer].append([pair[0], pair[0]])
                virtual_node_layer -= 1

    for i in range(len(layer_connections)):
        layer_connections[i] = np.array(layer_connections[i])
        layer_connections[i] = np.unique(layer_connections[i], axis=0)

    return layer_connections


def create_index(array: np.ndarray) -> dict:
    """Create a dictionary mapping unique elements to sequential integer indices.

    :param array: Array of elements (may contain duplicates).

    :returns: Dictionary mapping each unique element to an integer index.
    """
    unique_array = pd.unique(array)
    return {element: i for i, element in enumerate(unique_array)}


def _read_lines(source: str | Path | StringIO) -> list[str]:
    return (source.getvalue() if isinstance(source, StringIO) else Path(source).read_text()).splitlines()


def resolve_ontology_metadata(metadata: dict[str, Any]) -> tuple[list[np.ndarray], dict[str, int], list[str]]:
    """Resolve precomputed structures, file paths, or embedded ontology text.

    :param metadata: SparseGO dataset or feature-block metadata.
    :returns: Connections, gene indices and input column order.
    :raises ValueError: If ontology inputs or gene indices are incomplete.
    """
    connections = metadata.get("layer_connections")
    mapping = metadata.get("gene2id_mapping_ont")
    if connections is None or mapping is None:
        ontology = metadata.get("ontology_file")
        genes = metadata.get("gene2ind_file")
        if "ontology" in metadata and "gene2ind" in metadata:
            ontology, genes = StringIO(metadata["ontology"]), StringIO(metadata["gene2ind"])
        if ontology is None or genes is None:
            raise ValueError(
                "SparseGO metadata requires pre-computed ontology structures or ontology_file/gene2ind_file"
            )
        mapping = load_mapping(genes)
        graph, terms, annotations = load_ontology(ontology, mapping)
        pairs, levels, indices = sort_pairs(annotations, terms, graph, mapping)
        connections = pairs_in_layers(pairs, levels, indices)
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("SparseGO gene indices must be unique and contiguous from zero")
    order = metadata.get("ontology_gene_order")
    if order is None:
        order = sorted(mapping, key=mapping.get)
    if list(order) != sorted(mapping, key=mapping.get):
        raise ValueError("SparseGO ontology_gene_order must match gene indices")
    return list(connections), dict(mapping), list(order)
