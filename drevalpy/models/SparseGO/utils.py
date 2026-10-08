"""Utility functions for the SparseGO model.

Adapted from https://github.com/KatynaSada/SparseGO_lightning
"""

import sys

import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import numpy as np
import pandas as pd


def load_mapping(mapping_file: str) -> dict:
    """Open a txt file with two columns and return a dictionary.

    The second column becomes the key and the first column becomes the value.
    Used to read gene2ind.txt and drug2ind.txt files.

    :param mapping_file: Path to the mapping txt file.
    :return: Dictionary mapping names (e.g. gene symbols) to integer indices.
    """
    mapping = {}
    with open(mapping_file) as file_handle:
        for line in file_handle:
            line_parts = line.rstrip().split()
            mapping[line_parts[1]] = int(line_parts[0])
    return mapping


def load_ontology(ontology_file: str, gene2id_mapping: dict) -> tuple:
    """Create the directed graph of GO terms and store connected elements in arrays.

    The ontology file is tab-delimited with three columns:
    - Column 1: parent term or gene
    - Column 2: child term or gene
    - Column 3: 'default' for term-term connections, 'gene' for term-gene annotations

    :param ontology_file: Path to the ontology file (e.g. sparseGO_ont.txt).
    :param gene2id_mapping: Dictionary mapping gene names to integer IDs.
    :return: Tuple of (directed graph, term-term pairs array, gene-term pairs array).
    """
    dG = nx.DiGraph()
    terms_pairs: list[list[str]] = []
    genes_terms_pairs: list[list[str]] = []
    gene_set: set[str] = set()
    term_direct_gene_map: dict[str, set[int]] = {}
    term_size_map: dict[str, int] = {}

    with open(ontology_file) as file_handle:
        for line in file_handle:
            line_parts = line.rstrip().split()

            if line_parts[2] == "default":
                dG.add_edge(line_parts[0], line_parts[1])
                terms_pairs.append([line_parts[0], line_parts[1]])
            else:
                if line_parts[1] not in gene2id_mapping:
                    continue
                genes_terms_pairs.append([line_parts[0], line_parts[1]])
                if line_parts[0] not in term_direct_gene_map:
                    term_direct_gene_map[line_parts[0]] = set()
                term_direct_gene_map[line_parts[0]].add(gene2id_mapping[line_parts[1]])
                gene_set.add(line_parts[1])

    terms_pairs_arr: np.ndarray = np.array(terms_pairs)
    genes_terms_pairs_arr: np.ndarray = np.array(genes_terms_pairs)

    print(f"There are {len(gene_set)} genes")

    for term in dG.nodes():
        term_gene_set: set[int] = set()
        if term in term_direct_gene_map:
            term_gene_set = term_direct_gene_map[term]
        deslist = nxadag.descendants(dG, term)
        for child in deslist:
            if child in term_direct_gene_map:
                term_gene_set = term_gene_set | term_direct_gene_map[child]
        if len(term_gene_set) == 0:
            print(f"There is empty terms, please delete term: {term}")
            sys.exit(1)
        else:
            term_size_map[term] = len(term_gene_set)

    leaves = [n for n in dG.nodes if dG.in_degree(n) == 0]
    uG = dG.to_undirected()
    connected_subG_list = list(nxacc.connected_components(uG))

    print(f"There are {len(leaves)} roots: {leaves[0]}")
    print(f"There are {len(dG.nodes())} terms")
    print(f"There are {len(connected_subG_list)} connected components")

    if len(leaves) > 1:
        print("There are more than 1 root of ontology. Please use only one root.")
        sys.exit(1)
    if len(connected_subG_list) > 1:
        print("There are more than connected components. Please connect them.")
        sys.exit(1)

    return dG, terms_pairs_arr, genes_terms_pairs_arr


def sort_pairs(genes_terms_pairs: np.ndarray, terms_pairs: np.ndarray, dG: nx.DiGraph, gene2id_mapping: dict) -> tuple:
    """Concatenate and sort all pairs so the parent term is always in the first column.

    :param genes_terms_pairs: Array of (term, gene) pairs.
    :param terms_pairs: Array of (parent_term, child_term) pairs.
    :param dG: Directed graph of GO terms.
    :param gene2id_mapping: Dictionary mapping gene names to integer IDs.
    :return: Tuple of (sorted_pairs, level_list, level_number).
    """
    all_pairs = np.concatenate((genes_terms_pairs, terms_pairs))
    graph = dG.copy()

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
    :return: List of numpy arrays, one per layer, each containing (parent, child) pairs.
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
    :return: Dictionary mapping each unique element to an integer index.
    """
    unique_array = pd.unique(array)
    return {element: i for i, element in enumerate(unique_array)}
