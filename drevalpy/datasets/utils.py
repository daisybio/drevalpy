"""
Utility functions for datasets.
"""

import zipfile
import os
from typing import List
import requests
import numpy as np
from numpy.typing import ArrayLike
import networkx as nx


def download_dataset(
    dataset: str,
    data_path: str = "data",
    redownload: bool = False,
):
    """
    Download the latets dataset from Zenodo.
    :param dataset: dataset name, e.g., "GDSC1", "GDSC2" or "CCLE"
    :param data_path: where to save the data
    :param redownload: whether to redownload the data
    :return:
    """
    file_name = f"{dataset}.zip"
    file_path = os.path.join(data_path, file_name)
    if os.path.exists(file_path) and not redownload:
        print(f"{dataset} already exists, skipping download.")
    else:
        url = "https://zenodo.org/doi/10.5281/zenodo.12633909"
        # Fetch the latest record
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(
                f"Error fetching record: {response.status_code}"
            )
        latest_url = response.links["linkset"]["url"]
        response = requests.get(latest_url, timeout=10)
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(
                f"Error fetching record: {response.status_code}"
            )
        data = response.json()

        # Ensure the save path exists
        os.makedirs(data_path, exist_ok=True)

        # Download each file
        name_to_url = {file["key"]: file["links"]["self"] for file in data["files"]}
        file_url = name_to_url[file_name]
        # Download the file
        print(f"Downloading {dataset} from {file_url}...")
        response = requests.get(file_url, timeout=10)
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(
                f"Error downloading file {dataset}: " f"{response.status_code}"
            )

        # Save the file
        with open(file_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(file_path, "r") as z:
            for member in z.infolist():
                if not member.filename.startswith("__MACOSX/"):
                    z.extract(member, data_path)
        os.remove(file_path)  # Remove zip file after extraction

        print(f"{dataset} data downloaded and extracted to {data_path}")


def randomize_graph(original_graph: nx.Graph) -> nx.Graph:
    """
    Randomizes the graph by shuffling the edges while preserving the degree sequence.
    :param original_graph: The original graph
    :return: Randomized graph with the same degree sequence and node attributes
    """
    # Get the degree sequence from the original graph
    degree_sequence = [degree for node, degree in original_graph.degree()]

    # Generate a new graph with the expected degree sequence
    new_graph = nx.expected_degree_graph(degree_sequence, seed=1234)

    # Remap nodes to the original labels
    mapping = dict(zip(new_graph.nodes(), original_graph.nodes()))
    new_graph = nx.relabel_nodes(new_graph, mapping)

    # Copy node attributes from the original graph to the new graph
    for node, data in original_graph.nodes(data=True):
        new_graph.nodes[node].update(data)

    # Get the edge attributes from the original graph
    edge_attributes = list(original_graph.edges(data=True))

    # Assign random edge attributes to the new edges
    for edge in new_graph.edges():
        random_idx = int(np.random.randint(len(edge_attributes)))
        _, _, attr = edge_attributes[random_idx]
        new_graph[edge[0]][edge[1]].update(attr)

    return new_graph


def permute_features(
    features: dict, identifiers: ArrayLike, views_to_permute: List, all_views: List
) -> dict:
    """
    Permute the specified views for each entity (= cell line or drug)
    E.g. each cell line gets the feature vector/graph/image... of another cell line.
    Drawn without replacement.
    :param features: dictionary of features
    :param identifiers: array of identifiers
    :param views_to_permute: list of views to permute
    :param all_views: list of all views
    :return: permuted features
    """

    return {
        entity: {
            view: (
                features[entity][view]
                if view not in views_to_permute
                else features[other_entity][view]
            )
            for view in all_views
        }
        for entity, other_entity in zip(identifiers, np.random.permutation(identifiers))
    }
