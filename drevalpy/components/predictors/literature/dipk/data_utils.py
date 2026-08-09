"""Includes functions to load and process the DIPK dataset.

- get_data: Creates a list of dictionaries with drug and cell line features.
- CollateFn: Class to collate the DataLoader batches.
- DIPKDataset: Dataset class for the DIPK model.
"""

from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from drevalpy.components.feature_source import FeatureSource


def load_bionic_features(data_path: str | Path, dataset_name: str, gene_add_num: int = 512) -> dict[str, np.ndarray]:
    """Load biological network (BIONIC) features for DIPK.

    :param data_path: Path to the data, e.g., "data/"
    :param dataset_name: Name of the dataset, e.g., GDSC2
    :param gene_add_num: Number of genes to add to the feature set
    :returns: Dict mapping cell-line ID to BIONIC feature vector.
    """
    data_root = Path(data_path)
    gene_expression_path = data_root / dataset_name / "gene_expression.csv"
    gene_expression = pd.read_csv(gene_expression_path)
    expression_dict = gene_expression.set_index("cell_line_name").drop("cellosaurus_id", axis=1).T.to_dict()

    gene_list_path = data_root / dataset_name / "DIPK_features" / "gene_list_sel.txt"
    with open(gene_list_path, encoding="gbk") as f:
        gene_list = {line.strip() for line in f}

    ppi_path = data_root / dataset_name / "DIPK_features" / "human_ppi_features.tsv"
    dataset = pd.read_csv(ppi_path, index_col=0, sep="\t")

    bionic_gene_dict = {gene: dataset.loc[gene].values for gene in gene_list if gene in dataset.index}

    bionic_feature_dict: dict[str, np.ndarray] = {}
    for cell_line, expressions in expression_dict.items():
        sorted_genes = sorted(expressions.items(), key=lambda x: -x[1])
        top_genes = [gene for gene, _ in sorted_genes[:gene_add_num]]

        selected_features = [bionic_gene_dict[gene] for gene in top_genes if gene in bionic_gene_dict]
        if selected_features:
            aggregated_feature = np.mean(selected_features, axis=0)
        else:
            aggregated_feature = np.zeros(next(iter(bionic_gene_dict.values())).shape)

        bionic_feature_dict[cell_line] = aggregated_feature

    return bionic_feature_dict


def get_data(
    cell_ids: np.ndarray,
    drug_ids: np.ndarray,
    cell_line_source: FeatureSource,
    drug_source: FeatureSource,
    ic50: np.ndarray | None = None,
) -> list:
    """Prepare data samples for training or prediction.

    Each sample includes:

    - Drug features (e.g., molecular embeddings).
    - Cell line features (gene expression and bionic_features).
    - Optional IC50 response values for supervised tasks.

    :param cell_ids: IDs of the cell lines from the dataset.
    :param drug_ids: IDs of the drugs from the dataset.
    :param cell_line_source: FeatureSource providing cell-line views.
    :param drug_source: FeatureSource providing drug views.
    :param ic50: (Optional) Response values (e.g., IC50) to associate with samples.
    :returns: List of dictionaries, each containing drug and cell line features, with optional IC50.
    """
    data_list = []
    for i in range(len(cell_ids)):
        drug_id = str(drug_ids[i])
        cell_id = str(cell_ids[i])
        drug_tensor = torch.tensor(
            np.asarray(drug_source.get_entity_view(drug_id, "molgnet_features"), dtype=np.float32)
        )
        gene_expression = torch.tensor(
            np.asarray(cell_line_source.get_entity_view(cell_id, "gene_expression"), dtype=np.float32)
        )
        bionic_features = torch.tensor(
            np.asarray(cell_line_source.get_entity_view(cell_id, "bionic_features"), dtype=np.float32)
        )

        sample = {
            "molgnet_features": drug_tensor,
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
        """Initialize the CollateFn.

        :param train: indicates whether the DataLoader is used for training
        """
        self.train = train

    def __call__(self, batch):
        """Collate the batch.

        :param batch: batch of feature dictionaries
        :returns: collated node features, gene features, bionic features, and (optional) IC50 values
        """
        max_atoms_molgnet = max([sample["molgnet_features"].size(0) for sample in batch])

        padded_molgnet_features = []
        molgnet_mask = []

        for sample in batch:
            num_atoms = sample["molgnet_features"].size(0)
            padding_size = max_atoms_molgnet - num_atoms

            padded_features = torch.cat(
                [sample["molgnet_features"], torch.zeros(padding_size, sample["molgnet_features"].size(1))], dim=0
            )
            padded_molgnet_features.append(padded_features)

            mask = torch.cat(
                [torch.ones(num_atoms, dtype=torch.bool), torch.zeros(padding_size, dtype=torch.bool)], dim=0
            )
            molgnet_mask.append(mask)

        molgnet_features = torch.stack(padded_molgnet_features)
        molgnet_mask = torch.stack(molgnet_mask)

        gene_features = torch.stack([sample["gene_expression"] for sample in batch])
        bionic_features = torch.stack([sample["bionic_features"] for sample in batch])

        if self.train:
            ic50_values = torch.stack([sample["ic50"] for sample in batch])
            return {
                "molgnet_features": molgnet_features,
                "gene_features": gene_features,
                "bionic_features": bionic_features,
                "ic50_values": ic50_values,
                "molgnet_mask": molgnet_mask,
            }
        else:
            return {
                "molgnet_features": molgnet_features,
                "gene_features": gene_features,
                "bionic_features": bionic_features,
                "molgnet_mask": molgnet_mask,
            }


class DIPKDataset(Dataset, ABC):
    """Dataset of graphs from get_data."""

    def __init__(self, samples):
        """Initialize the GraphDataset.

        :param samples: list
        """
        super().__init__()
        self._samples = samples

    def __getitem__(self, idx):
        """Get the sample at index idx.

        :param idx: index
        :returns: sample
        """
        sample = self._samples[idx]
        return sample

    def __len__(self) -> int:
        """Get the number of graphs in the dataset.

        :returns: number of samples
        """
        return len(self._samples)
