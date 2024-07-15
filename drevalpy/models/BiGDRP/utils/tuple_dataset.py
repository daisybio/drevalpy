import torch
from torch.utils.data import Dataset


class TupleMatrixDataset(Dataset):
    def __init__(
        self, tuples, cell_features, label_matrix, bin_label_matrix=None, weighted=False
    ):
        order = ["drug", "cell_line"]
        self.tuples = tuples[order].values
        self.cell_features = cell_features
        self.label_matrix = label_matrix
        self.num_drugs = label_matrix.shape[1]
        self.bin_label_matrix = bin_label_matrix

        if weighted:
            self.weight = 1.0 / label_matrix.std(axis=0)
        else:
            self.weight = torch.ones(self.num_drugs)

        if self.label_matrix.dim() == 2:
            self.label_matrix = self.label_matrix.unsqueeze(2)

        if self.bin_label_matrix is not None and self.bin_label_matrix.dim() == 2:
            self.bin_label_matrix = self.bin_label_matrix.unsqueeze(2)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.to_list()

        tuples = self.tuples[idx]
        drug = tuples[0]
        cell = tuples[1]

        if self.bin_label_matrix is None:
            sample = (
                self.cell_features[cell],
                drug,
                self.label_matrix[cell, drug],
                self.weight[drug],
            )
        else:
            sample = (
                self.cell_features[cell],
                drug,
                self.label_matrix[cell, drug],
                self.bin_label_matrix[cell, drug],
                self.weight[drug],
            )

        return sample

    def get_all_labels(self):
        return self.labels.numpy()


class TupleMapDataset(Dataset):
    def __init__(
        self, tuples, drug_features, cell_features, label_matrix, bin_label_matrix=None
    ):

        order = ["drug", "cell_line"]
        self.tuples = tuples[order].values  # index = range(len(triplets))
        self.cell_features = torch.Tensor(cell_features)
        self.drug_features = torch.Tensor(drug_features.values)
        self.label_matrix = torch.Tensor(label_matrix)
        self.num_drugs = label_matrix.shape[1]

        if self.label_matrix.dim() == 2:
            self.label_matrix = self.label_matrix.unsqueeze(2)

        self.bin_label_matrix = bin_label_matrix
        if bin_label_matrix is not None:
            self.bin_label_matrix = torch.Tensor(bin_label_matrix)
            if self.bin_label_matrix.dim() == 2:
                self.bin_label_matrix = self.bin_label_matrix.unsqueeze(2)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx.to_list()

        tuples = self.tuples[idx]
        drug = tuples[0]
        cell = tuples[1]

        if self.bin_label_matrix is None:
            sample = (
                self.cell_features[cell],
                self.drug_features[drug],
                self.label_matrix[cell, drug],
            )
        else:
            sample = (
                self.cell_features[cell],
                self.drug_features[drug],
                self.label_matrix[cell, drug],
                self.bin_label_matrix[cell, drug],
            )

        return sample
