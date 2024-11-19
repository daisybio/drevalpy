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
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data, Dataset

from drevalpy.datasets.dataset import FeatureDataset


def load_bionic_features(data_path: str, dataset_name: str, gene_add_num: int = 512) -> FeatureDataset:
    """Load biological network (BIONIC) features for DIPK.

    :param data_path: Path to the data, e.g., "data/"
    :param dataset_name: Name of the dataset, e.g., GDSC2
    :param gene_add_num: Number of genes to add to the feature set
    :returns: FeatureDataset with gene expression and biological network features
    """
    # Load gene expression dataset and extract gene names
    gene_expression = pd.read_csv(f"{data_path}/{dataset_name}/gene_expression.csv")
    expression_dict = gene_expression.set_index("cell_line_name").drop("cellosaurus_id", axis=1).T.to_dict()

    f = open("data/GDSC1/DIPK_features/gene_list_sel.txt", encoding="gbk")
    gene_list = []
    for each_row in f:
        gene_list.append(each_row.strip())

    bionic_gene_dict = dict()
    dataset = pd.read_csv("data/GDSC1/DIPK_features/human_ppi_features.tsv", header=0, index_col=0, sep="\t")
    for gene in gene_list:
        if gene in dataset.index:
            bionic_gene_dict[gene] = dataset.loc[gene].values

    cells = list(expression_dict.keys())  # List of cell line names

    # Sort gene expressions in descending order for each cell line
    indices_map = {}
    for cell_line in cells:
        indices = np.argsort(cell_line)[::-1]
        indices_map[cell_line] = indices

    # Compute BIONIC features for each cell line
    bionic_feature_dict = {}
    for cell in cells:

        selected_genes = indices_map[cell][:gene_add_num].tolist()  # Top `gene_add_num` genes for this cell line
        selected_features = [bionic_gene_dict[gene_id] for gene_id in selected_genes if gene_id in bionic_gene_dict]

        # Check if any valid features are found
        if selected_features:
            feature_tensor = np.stack(selected_features)  # Stack into a 2D tensor (genes x feature_dim)
            aggregated_feature = feature_tensor.mean(dim=0)  # Compute the mean feature across genes
        else:
            aggregated_feature = np.zeros(
                len(next(iter(bionic_gene_dict.values())))
            )  # Fill with zeros if no valid features are found

        # Store the aggregated feature in the dictionary
        bionic_feature_dict[cell] = aggregated_feature.tolist()

    # Structure data into a FeatureDataset
    feature_data = defaultdict(dict)
    for cell_line in cells:
        feature_data[cell_line]["bionic_features"] = bionic_feature_dict[cell_line]

    return FeatureDataset(features=feature_data)


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
