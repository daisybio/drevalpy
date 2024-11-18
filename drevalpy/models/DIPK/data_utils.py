"""
Includes functions to load and process the DIPK dataset.

load_expression_and_network_features: Loads gene expression and biological network features from the DIPK dataset.
load_drug_feature_from_MolGNet: Loads drug features from the MolGNet dataset.
get_data: Creates a list of PyG Data objects from the input cell line and drug features.
CollateFn: Class to collate PyG Data objects for the DataLoader.
GraphDataset: Class to create a PyG Dataset from a list of PyG Data objects.
"""

import os
from abc import ABC

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data, Dataset

from drevalpy.datasets.dataset import FeatureDataset


def load_expression_and_network_features(
    feature_type1: str, feature_type2: str, data_path: str, dataset_name: str
) -> FeatureDataset:
    """
    Load gene expression and biological network features from the DIPK dataset.

    :param feature_type1: gene_expression_features
    :param feature_type2: biological_network_features
    :param data_path: path to the data, e.g. "data/"
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with gene expression and biological network features
    """
    expression_path = os.path.join(data_path, dataset_name, "DIPK_features", "GEF.csv")
    network_path = os.path.join(data_path, dataset_name, "DIPK_features", "BNF.csv")
    expression = pd.read_csv(expression_path, index_col=0)
    network = pd.read_csv(network_path, index_col=0, sep="\t")

    return FeatureDataset(
        features={
            celllines: {
                feature_type1: np.array(expression.loc[celllines].values.astype(float)),
                feature_type2: np.array(network.loc[celllines].values.astype(float)),
            }
            for celllines in expression.index
        }
    )


def load_drug_feature_from_mol_g_net(
    feature_type: str,
    feature_subtype1: str,
    feature_subtype2: str,
    feature_subtype3: str,
    data_path: str,
    dataset_name: str,
) -> FeatureDataset:
    """
    Load drug features from the MolGNet dataset.

    :param feature_type: drug_feature_embedding
    :param feature_subtype1: MolGNet_features
    :param feature_subtype2: Edge_Index
    :param feature_subtype3: Edge_Attr
    :param data_path: path to the data, e.g. "data/"
    :param dataset_name: name of the dataset, e.g., GDSC2
    :returns: FeatureDataset with drug features
    """

    def load_feature(file_path, sep="\t"):
        return np.array(pd.read_csv(file_path, index_col=0, sep=sep))

    drug_path = os.path.join(data_path, dataset_name, "DIPK_features", "Drugs")
    drug_list = os.listdir(drug_path)

    return FeatureDataset(
        features={
            drug: {
                feature_type: {
                    feature_subtype1: load_feature(os.path.join(drug_path, drug, f"MolGNet_{drug}.csv")),
                    feature_subtype2: load_feature(os.path.join(drug_path, drug, f"Edge_Index_{drug}.csv")),
                    feature_subtype3: load_feature(os.path.join(drug_path, drug, f"Edge_Attr_{drug}.csv")),
                }
            }
            for drug in drug_list
        }
    )


def get_data(
    cell_ids: np.ndarray,
    drug_ids: np.ndarray,
    cell_line_features: FeatureDataset,
    drug_features: FeatureDataset,
    ic50: np.ndarray | None = None,
):
    """
    Create a list of PyG Data objects from the input cell line and drug features.

    :param cell_ids: ids of the cell lines of the DrugResponseDataset
    :param drug_ids: ids of the drugs of the DrugResponseDataset
    :param cell_line_features: input cell line features
    :param drug_features: input drug features
    :param ic50: response values from the DrugResponseDataset
    :returns: list of PyG Data objects
    """
    graph_list = []
    for i in range(len(cell_ids)):
        drug_id = str(drug_ids[i])
        cell_id = str(cell_ids[i])
        x = torch.tensor(
            drug_features.features[drug_id]["drug_feature_embedding"]["MolGNet_features"], dtype=torch.float32
        )
        edge_index = torch.tensor(
            drug_features.features[drug_id]["drug_feature_embedding"]["Edge_Index"], dtype=torch.float32
        )
        edge_attr = torch.tensor(
            drug_features.features[drug_id]["drug_feature_embedding"]["Edge_Attr"], dtype=torch.float32
        )
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            GEF=torch.tensor(cell_line_features.features[cell_id]["gene_expression"], dtype=torch.float32),
            BNF=torch.tensor(cell_line_features.features[cell_id]["biological_network_features"], dtype=torch.float32),
        )
        if ic50 is not None:
            graph_data.ic50 = torch.tensor([ic50[i]], dtype=torch.float32)
        graph_list.append(graph_data)

    return graph_list


class CollateFn:
    """Collate function for the DataLoader, either for training or testing."""

    def __init__(self, train=True, follow_batch=None, exclude_keys=None):
        """
        Initialize the CollateFn.

        :param train: indicates whether the DataLoader is used for training
        :param follow_batch: unused
        :param exclude_keys: unused
        """
        self.train = train
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        """
        Collate the batch.

        :param batch: batch of PyG Data objects
        :returns: PyG Batch, gene features, and bionic features
        """
        pyg_list = [Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr) for g in batch]
        if self.train:
            for g, data in zip(batch, pyg_list):
                data.ic50 = g.ic50

        pyg_batch = Batch.from_data_list(pyg_list, self.follow_batch, self.exclude_keys)
        gene_features = torch.stack([g.GEF for g in batch])
        bionic_features = torch.stack([g.BNF for g in batch])

        return pyg_batch, gene_features, bionic_features


class GraphDataset(Dataset, ABC):
    """Dataset of graphs from get_data."""

    def __init__(self, graphs):
        """
        Initialize the GraphDataset.

        :param graphs: list of PyG Data objects
        """
        super().__init__()
        self._graphs = graphs

    def __getitem__(self, idx):
        """
        Get the graph at index idx.

        :param idx: index
        :returns: PyG Data object
        """
        graph = self._graphs[idx]
        return graph

    def __len__(self) -> int:
        """
        Get the number of graphs in the dataset.

        :return: number of graphs
        """
        return len(self._graphs)
