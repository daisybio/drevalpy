"""Utility functions for datasets."""

import zipfile
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import requests


def download_dataset(
    dataset_name: str,
    data_path: str | Path = "data",
    redownload: bool = False,
):
    """
    Download the latets dataset from Zenodo.

    :param dataset_name: dataset name, e.g., "GDSC1", "GDSC2", "CCLE" or "Toy_Data"
    :param data_path: where to save the data
    :param redownload: whether to redownload the data
    :raises HTTPError: if the download fails
    """
    file_name = f"{dataset_name}.zip"
    file_path = Path(data_path) / file_name
    extracted_folder_path = file_path.with_suffix("")

    # Check if the extracted data exists and skip download if not redownloading
    if extracted_folder_path.exists() and not redownload:
        print(f"{dataset_name} is already extracted, skipping download.")
    else:
        url = "https://zenodo.org/doi/10.5281/zenodo.12633909"
        # Fetch the latest record
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(f"Error fetching record: {response.status_code}")
        latest_url = response.links["linkset"]["url"]
        response = requests.get(latest_url, timeout=60)
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(f"Error fetching record: {response.status_code}")
        data = response.json()

        # Ensure the save path exists
        extracted_folder_path.parent.mkdir(exist_ok=True, parents=True)

        # Download each file
        name_to_url = {file["key"]: file["links"]["self"] for file in data["files"]}
        file_url = name_to_url[file_name]
        # Download the file
        print(f"Downloading {dataset_name} from {file_url}...")
        response = requests.get(file_url, timeout=60)
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(f"Error downloading file {dataset_name}: " f"{response.status_code}")

        # Save the file
        with open(file_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(file_path, "r") as z:
            for member in z.infolist():
                if not member.filename.startswith("__MACOSX/"):
                    z.extract(member, data_path)
        file_path.unlink()  # Remove zip file after extraction

        print(f"{dataset_name} data downloaded and extracted to {data_path}")


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
    mapping = dict(zip(new_graph.nodes(), original_graph.nodes(), strict=True))
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
    features: dict[str, dict[str, Any]],
    identifiers: np.ndarray,
    views_to_permute: list[str],
    all_views: list[str],
) -> dict:
    """
    Permute the specified views for each entity (= cell line or drug).

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
            view: (features[entity][view] if view not in views_to_permute else features[other_entity][view])
            for view in all_views
        }
        for entity, other_entity in zip(identifiers, np.random.permutation(identifiers), strict=True)
    }
