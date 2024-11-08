from abc import ABC
import pandas as pd
import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from drevalpy.datasets.dataset import FeatureDataset
import os
import numpy as np


def load_expression_and_network_features(
    feature_type1: str, feature_type2: str, data_path: str, dataset_name: str
) -> FeatureDataset:
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


def load_drug_feature_from_MolGNet(
    feature_type: str,
    feature_subtype1: str,
    feature_subtype2: str,
    feature_subtype3: str,
    data_path: str,
    dataset_name: str,
) -> FeatureDataset:
    
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


def get_data(cell_id, drug_id, cell_line_features, drug_features, ic50=None):

    graph_list = []
    for i in range(len(cell_id)):
        x = torch.tensor(
            drug_features.features[drug_id[i]]["drug_feature_embedding"]["MolGNet_features"], dtype=torch.float32
        )
        edge_index = torch.tensor(
            drug_features.features[drug_id[i]]["drug_feature_embedding"]["Edge_Index"], dtype=torch.float32
        )
        edge_attr = torch.tensor(
            drug_features.features[drug_id[i]]["drug_feature_embedding"]["Edge_Attr"], dtype=torch.float32
        )
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            GEF=torch.tensor(cell_line_features.features[cell_id[i]]["gene_expression_features"], dtype=torch.float32),
            BNF=torch.tensor(cell_line_features.features[cell_id[i]]["biological_network_features"], dtype=torch.float32),
        )
        if ic50 is not None:
            graph_data.ic50 = torch.tensor([ic50[i]], dtype=torch.float32)
        graph_list.append(graph_data)

    return graph_list


class CollateFn:
    def __init__(self, train=True, follow_batch=None, exclude_keys=None):
        self.train = train
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        pyg_list = [Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr) for g in batch]
        if self.train:
            for g, data in zip(batch, pyg_list):
                data.ic50 = g.ic50

        pyg_batch = Batch.from_data_list(pyg_list, self.follow_batch, self.exclude_keys)
        gene_features = torch.stack([g.GEF for g in batch])
        bionic_features = torch.stack([g.BNF for g in batch])

        return pyg_batch, gene_features, bionic_features


class GraphDataset(Dataset, ABC):
    def __init__(self, graphs):
        self._graphs = graphs

    def __getitem__(self, idx):
        graph = self._graphs[idx]
        return graph

    def __len__(self):
        return len(self._graphs)
