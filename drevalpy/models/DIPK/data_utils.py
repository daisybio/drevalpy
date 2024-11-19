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
from torch.utils.data import Dataset

from drevalpy.datasets.dataset import FeatureDataset


def load_bionic_features(data_path: str, dataset_name: str, gene_add_num: int = 512) -> FeatureDataset:
    """
    Load biological network (BIONIC) features for DIPK.

    :param data_path: Path to the data, e.g., "data/"
    :param dataset_name: Name of the dataset, e.g., GDSC2
    :param gene_add_num: Number of genes to add to the feature set
    :returns: FeatureDataset with gene expression and biological network features
    """
    # Load gene expression dataset
    gene_expression_path = os.path.join(data_path, dataset_name, "gene_expression.csv")
    gene_expression = pd.read_csv(gene_expression_path)
    expression_dict = gene_expression.set_index("cell_line_name").drop("cellosaurus_id", axis=1).T.to_dict()

    # Load gene list and PPI features
    gene_list_path = os.path.join(data_path, dataset_name, "DIPK_features", "gene_list_sel.txt")
    with open(gene_list_path, encoding="gbk") as f:
        gene_list = {line.strip() for line in f}

    ppi_path = os.path.join(data_path, dataset_name, "DIPK_features", "human_ppi_features.tsv")
    dataset = pd.read_csv(ppi_path, index_col=0, sep="\t")

    # Ensure BIONIC dictionary uses gene names directly
    bionic_gene_dict = {gene: dataset.loc[gene].values for gene in gene_list if gene in dataset.index}

    # Compute BIONIC features
    bionic_feature_dict = {}
    for cell_line, expressions in expression_dict.items():
        # Sort genes based on descending expression values
        sorted_genes = sorted(expressions.items(), key=lambda x: -x[1])
        top_genes = [gene for gene, _ in sorted_genes[:gene_add_num]]

        # Aggregate BIONIC features for selected genes
        selected_features = [bionic_gene_dict[gene] for gene in top_genes if gene in bionic_gene_dict]
        if selected_features:
            aggregated_feature = np.mean(selected_features, axis=0)
        else:
            # Handle case where no features are found (padding with zeros)
            aggregated_feature = np.zeros(next(iter(bionic_gene_dict.values())).shape)

        bionic_feature_dict[cell_line] = aggregated_feature

    feature_data = {cell_line: {"bionic_features": features} for cell_line, features in bionic_feature_dict.items()}
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
    drug_list = [drug for drug in os.listdir(drug_path) if drug != ".DS_Store"]  # .DS_Store is a macOS file, ignore it

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
) -> list:
    """
    Prepare data samples for training or prediction.

    Each sample includes:
    - Drug features (e.g., molecular embeddings).
    - Cell line features (gene expression and biological network features).
    - Optional IC50 response values for supervised tasks.

    :param cell_ids: IDs of the cell lines from the dataset.
    :param drug_ids: IDs of the drugs from the dataset.
    :param cell_line_features: Input features associated with the cell lines.
    :param drug_features: Input features associated with the drugs.
    :param ic50: (Optional) Response values (e.g., IC50) to associate with samples.
    :return: List of dictionaries, each containing drug and cell line features, with optional IC50.
    """
    data_list = []
    for i in range(len(cell_ids)):
        drug_id = str(drug_ids[i])
        cell_id = str(cell_ids[i])
        drug_tensor = torch.tensor(
            drug_features.features[drug_id]["drug_feature_embedding"]["MolGNet_features"], dtype=torch.float32
        )
        gene_expression = torch.tensor(cell_line_features.features[cell_id]["gene_expression"], dtype=torch.float32)
        bionic_features = torch.tensor(
            cell_line_features.features[cell_id]["biological_network_features"], dtype=torch.float32
        )

        sample = {
            "drug_features": drug_tensor,
            "gene_expression": gene_expression,
            "bionic_features": bionic_features,
        }
        if ic50 is not None:
            sample["ic50"] = torch.tensor([ic50[i]], dtype=torch.float32)

        data_list.append(sample)

    return data_list


class CollateFn:
    """Collate function for the DataLoader, either for training or testing."""

    def __init__(self, train=True):
        """
        Initialize the CollateFn.

        :param train: indicates whether the DataLoader is used for training
        """
        self.train = train

    def __call__(self, batch):
        """
        Collate the batch.

        :param batch: batch of feature dictionaries
        :returns: collated node features, gene features, bionic features, and (optional) IC50 values
        """
        drug_features = torch.stack([sample["drug_features"] for sample in batch])
        gene_features = torch.stack([sample["gene_expression"] for sample in batch])
        bionic_features = torch.stack([sample["bionic_features"] for sample in batch])

        if self.train:
            ic50_values = torch.stack([sample["ic50"] for sample in batch])
            return drug_features, gene_features, bionic_features, ic50_values
        else:
            return drug_features, gene_features, bionic_features


class DIPKDataset(Dataset, ABC):
    """Dataset of graphs from get_data."""

    def __init__(self, samples):
        """
        Initialize the GraphDataset.

        :param samples: list
        """
        super().__init__()
        self._samples = samples

    def __getitem__(self, idx):
        """
        Get the sample at index idx.

        :param idx: index
        :returns: sample
        """
        sample = self._samples[idx]
        return sample

    def __len__(self) -> int:
        """
        Get the number of graphs in the dataset.

        :return: number of samples
        """
        return len(self._samples)
