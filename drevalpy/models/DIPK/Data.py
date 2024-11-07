from abc import ABC
import pandas as pd
import joblib
import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
import os

import numpy as np
import pandas as pd

def load_expression_and_network_features(feature_type1: str, feature_type2: str, data_path: str, dataset_name: str) -> FeatureDataset:
    expression = pd.read_csv(f"{data_path}/{dataset_name}/DIPK_features/GEF.csv", index_col=0)
    network = pd.read_csv(f"{data_path}/{dataset_name}/DIPK_features/BNF.csv", index_col=0, sep='\t')

    return FeatureDataset(
        features={celllines: {feature_type1: np.array(expression.loc[celllines].values.astype(float)), feature_type2: np.array(network.loc[celllines].values.astype(float))} for celllines in expression.index}
    )

def load_drug_feature_from_MolGNet(feature_type: str, feature_subtype1: str, feature_subtype2: str, feature_subtype3: str, data_path: str, dataset_name: str) -> FeatureDataset:
    drug_list = os.listdir(f"{data_path}/{dataset_name}/DIPK_features/Drugs")
    
    return FeatureDataset(
        features={drugs: {feature_type: {feature_subtype1: np.array(pd.read_csv(f"{data_path}/{dataset_name}/DIPK_features/Drugs/{drugs}/MolGNet_{drugs}.csv", index_col=0, sep='\t')), feature_subtype2: np.array(pd.read_csv(f"{data_path}/{dataset_name}/DIPK_features/Drugs/{drugs}/Edge_Index_{drugs}.csv", index_col=0, sep='\t')), feature_subtype3: np.array(pd.read_csv(f"{data_path}/{dataset_name}/DIPK_features/Drugs/{drugs}/Edge_Attr_{drugs}.csv", index_col=0, sep='\t'))}} for drugs in drug_list}
    )

def GetTestData(cell_id, drug_id, cell_line_features, drug_features):
    
    Cell = cell_id
    Drug = drug_id
    
    Graph = []
    for ii in range(len(Cell)):
        x = torch.tensor(drug_features.features[Drug[ii]]["drug_feature_embedding"]["MolGNet_features"],  dtype=torch.float32)
        edge_index = torch.tensor(drug_features.features[Drug[ii]]["drug_feature_embedding"]["Edge_Index"],  dtype=torch.float32)
        edge_attr = torch.tensor(drug_features.features[Drug[ii]]["drug_feature_embedding"]["Edge_Attr"],  dtype=torch.float32)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    GEF=torch.tensor(cell_line_features.features[Cell[ii]]["gene_expression_features"], dtype=torch.float32),
                    BNF=torch.tensor(cell_line_features.features[Cell[ii]]["biological_network_features"], dtype=torch.float32))
        Graph.append(graph)

    return Graph
 
def GetTrainData(cell_id, drug_id, ic50, cell_line_features, drug_features):    
    
    Cell = cell_id
    Drug = drug_id
    IC50 = ic50

    Graph = []
    for ii in range(len(Cell)):
        x = torch.tensor(drug_features.features[Drug[ii]]["drug_feature_embedding"]["MolGNet_features"], dtype=torch.float32)
        edge_index = torch.tensor(drug_features.features[Drug[ii]]["drug_feature_embedding"]["Edge_Index"], dtype=torch.float32)
        edge_attr = torch.tensor(drug_features.features[Drug[ii]]["drug_feature_embedding"]["Edge_Attr"], dtype=torch.float32)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    GEF=torch.tensor(cell_line_features.features[Cell[ii]]["gene_expression_features"], dtype=torch.float32),
                    BNF=torch.tensor(cell_line_features.features[Cell[ii]]["biological_network_features"], dtype=torch.float32),
                    ic50=torch.tensor([IC50[ii]], dtype=torch.float32))
        Graph.append(graph)

    return Graph

class CollateFn_Train:
    def __init__(self, follow_batch=None, exclude_keys=None):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        pyg_list = [Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, ic50=g.ic50) for g in batch]
        pyg_batch = Batch.from_data_list(pyg_list, self.follow_batch, self.exclude_keys)
        GeneFt = torch.stack([g.GEF for g in batch])
        BionicFt = torch.stack([g.BNF for g in batch])
        
        return pyg_batch, GeneFt, BionicFt
    
class CollateFn_Test:
    def __init__(self, follow_batch=None, exclude_keys=None):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        pyg_list = [Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr) for g in batch]
        pyg_batch = Batch.from_data_list(pyg_list, self.follow_batch, self.exclude_keys)
        GeneFt = torch.stack([g.GEF for g in batch])
        BionicFt = torch.stack([g.BNF for g in batch])
        
        return pyg_batch, GeneFt, BionicFt
 
class MyDataSet(Dataset, ABC):
    def __init__(self, graphs):
        self._graphs = graphs

    def __getitem__(self, idx):
        graph = self._graphs[idx]
        return graph

    def __len__(self):
        return len(self._graphs)
