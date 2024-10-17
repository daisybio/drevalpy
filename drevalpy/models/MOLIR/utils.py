"""
Code for the MOLI model.
Original authors: Sharifi-Noghabi et al. (2019, 10.1093/bioinformatics/btz318)
Code adapted from: Hauptmann et al. (2023, 10.1186/s12859-023-05166-7), https://github.com/kramerlab/Multi-Omics_analysis
"""

import subprocess
from io import BytesIO
import pandas as pd
from typing import Optional, List, Tuple, Dict
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
import random
import numpy as np
from itertools import combinations


class RegressionDataset(Dataset):
    """
    Dataset for regression tasks for the data loader.
    """

    def __init__(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset = None,
    ):
        self.output = output
        self.cell_line_input = cell_line_input

    def __getitem__(self, idx):
        response = self.output.response[idx].astype(np.float32)

        cell_line_id = self.output.cell_line_ids[idx]
        gene_expression = self.cell_line_input.features[cell_line_id]["gene_expression"].astype(np.float32)
        mutations = self.cell_line_input.features[cell_line_id]["mutations"].astype(np.float32)
        copy_number = self.cell_line_input.features[cell_line_id]["copy_number_variation_gistic"].astype(np.float32)

        return gene_expression, mutations, copy_number, response

    def __len__(self):
        "Overwrites the len method."
        return len(self.output.response)


def generate_triplets_indices(
y: np.ndarray,
positive_range: float,
negative_range: float,
random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates triplets for the MOLI model.
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    positive_sample_indices = []
    negative_sample_indices = []
    # Iterate over each label in the dataset
    for idx_current_label, current_label in enumerate(y):
        positive_class_indices = get_positive_class_indices(
          current_label, idx_current_label, y, positive_range
        )
        positive_sample_idx = np.random.choice(positive_class_indices, 1)[0]
        negative_class_indices = get_negative_class_indices(
          current_label, y, negative_range
        )
        negative_sample_idx = np.random.choice(negative_class_indices, 1)[0]
        positive_sample_indices.append(positive_sample_idx)
        negative_sample_indices.append(negative_sample_idx)
    return np.array(positive_sample_indices), np.array(negative_sample_indices)


def get_positive_class_indices(
    label: float, idx_label: int, y: np.ndarray, positive_range: float
) -> np.ndarray:
    # find the samples that are within the positive range of the label except the label itself
    indices_similar_samples = np.where(
        np.logical_and(label - positive_range <= y, y <= label + positive_range)
    )[0]
    indices_similar_samples = np.delete(indices_similar_samples,
                                        np.where(indices_similar_samples == idx_label)
                                        )
    if len(indices_similar_samples) == 0:
        # return the closest samples to the label except the label itself
        indices_similar_samples = np.array([np.argsort(np.abs(y - label))[1]])
    return indices_similar_samples


def get_negative_class_indices(
    label: float, y: np.ndarray, negative_range: float
) -> np.ndarray:
    dissimilar_samples = np.where(
        np.logical_or(label - negative_range >= y, y >= label + negative_range)
    )[0]
    if len(dissimilar_samples) == 0:
        # return the sample that is the furthest away from the label
        dissimilar_samples = np.argsort(np.abs(y - label))[-1:]
    return dissimilar_samples


def generate_anchor_positive_pairs(
    positive_class_indices: np.ndarray, num_pairs: int
) -> List[Tuple[int, int]]:
    if len(positive_class_indices) <= 1:
        raise ValueError("Not enough positive samples to generate pairs.")
    if len(positive_class_indices) <= 15:
        return list(combinations(positive_class_indices, 2))
    else:
        sampled_indices = random.sample(list(positive_class_indices), k=15)
        return random.sample(list(combinations(sampled_indices, 2)), k=num_pairs)


def generate_negative_samples(
    negative_class_indices: np.ndarray, num_samples: int
) -> List[int]:
    if len(negative_class_indices) <= 1:
        raise ValueError("Not enough negative samples to generate pairs.")
    if len(negative_class_indices) <= 15:
        return list(negative_class_indices)
    else:
        return random.sample(list(negative_class_indices), k=num_samples)


class MOLIEncoder(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(MOLIEncoder, self).__init__()
        self.encode = torch.nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.encode(x)


class TripletDataset(Dataset):
    def __init__(self, triplets: Dict[str, np.ndarray], labels: np.ndarray):
        """
        Custom Dataset for PyTorch to handle triplets of features and their corresponding labels.

        Parameters:
        -----------
        triplets : Dict[str, np.ndarray]
            A dictionary containing the triplets for each feature type ('x_gene_expression',
            'x_mutations', 'x_copy_number_variation_gistic').
        labels : np.ndarray
            Corresponding labels for the triplets.
        """
        self.x_gene_expression = triplets["x_gene_expression"]
        self.x_mutations = triplets["x_mutations"]
        self.x_copy_number_variation_gistic = triplets["x_copy_number_variation_gistic"]
        self.labels = labels

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Fetches a sample for a given index."""
        x_e_triplet = self.x_gene_expression[idx]
        x_m_triplet = self.x_mutations[idx]
        x_c_triplet = self.x_copy_number_variation_gistic[idx]
        label = self.labels[idx]

        # Convert to PyTorch tensors
        return (
            torch.tensor(x_e_triplet, dtype=torch.float32),
            torch.tensor(x_m_triplet, dtype=torch.float32),
            torch.tensor(x_c_triplet, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


class MOLIRegressor(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(MOLIRegressor, self).__init__()
        self.classifier = torch.nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, 1),
        )

    def forward(self, x):
        return self.classifier(x)


class Moli(nn.Module):
    def __init__(self, hpams):
        super().__init__()
        self.mini_batch = hpams["mini_batch"]
        self.h_dim1 = hpams["h_dim1"]
        self.h_dim2 = hpams["h_dim2"]
        self.h_dim3 = hpams["h_dim3"]
        self.lr_e = hpams["lr_e"]
        self.lr_m = hpams["lr_m"]
        self.lr_c = hpams["lr_c"]
        self.lr_regr = hpams["lr_regr"]
        self.dropout_rate = hpams["dropout_rate"]
        self.weight_decay = hpams["weight_decay"]
        self.gamma = hpams["gamma"]
        self.epochs = hpams["epochs"]
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=hpams["margin"], p=2)
        self.regression_loss = nn.MSELoss()
        self.model_initialized = False
        self.expression_encoder = None
        self.mutation_encoder = None
        self.cna_encoder = None
        self.regressor = None
        self.positive_range = None
        self.negative_range = None

    def initialize_model(self, x_train_e, x_train_m, x_train_c):
        _, ie_dim = x_train_e.shape
        _, im_dim = x_train_m.shape
        _, ic_dim = x_train_c.shape
        self.expression_encoder = MOLIEncoder(ie_dim, self.h_dim1, self.dropout_rate)
        self.mutation_encoder = MOLIEncoder(im_dim, self.h_dim2, self.dropout_rate)
        self.cna_encoder = MOLIEncoder(ic_dim, self.h_dim3, self.dropout_rate)
        self.regressor = MOLIRegressor(self.h_dim1 + self.h_dim2 + self.h_dim3, self.dropout_rate)
        self.model_initialized = True

    def fit(
        self,
        output_train: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        output_earlystopping: Optional[DrugResponseDataset] = None,
        patience: int = 5,
    ):
        device = create_device(gpu_number=None)

        self.positive_range = np.std(output_train.response) * 0.1
        self.negative_range = np.std(output_train.response)
        train_dataset = RegressionDataset(
            output=output_train,
            cell_line_input=cell_line_input
        )

        # Create the DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.mini_batch,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

        val_loader = None
        if output_earlystopping is not None:
            val_dataset = RegressionDataset(
                output=output_earlystopping,
                cell_line_input=cell_line_input,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.mini_batch,
                shuffle=False,
                num_workers=1,
                persistent_workers=True,
            )

        self.force_initialize(train_loader)

        moli_optimiser = torch.optim.Adagrad(
            [
                {"params": self.expression_encoder.parameters(), "lr": self.lr_e},
                {"params": self.mutation_encoder.parameters(), "lr": self.lr_m},
                {"params": self.cna_encoder.parameters(), "lr": self.lr_c},
                {"params": self.regressor.parameters(), "lr": self.lr_regr},
            ],
            weight_decay=self.weight_decay,
        )
        last_val_loss = None
        for epoch in range(self.epochs):
            epoch_train_loss = 0
            for (x_train_e, x_train_m, x_train_c, y_train) in train_loader:
                x_train_e = x_train_e.to(device)
                x_train_m = x_train_m.to(device)
                x_train_c = x_train_c.to(device)
                y_train = y_train.to(device)
                self.expression_encoder = self.expression_encoder.to(device)
                self.mutation_encoder = self.mutation_encoder.to(device)
                self.cna_encoder = self.cna_encoder.to(device)
                self.regressor = self.regressor.to(device)

                self.cna_encoder.train()
                self.mutation_encoder.train()
                self.expression_encoder.train()
                self.regressor.train()

                z_ex = self.expression_encoder(x_train_e)
                z_mu = self.mutation_encoder(x_train_m)
                z_cn = self.cna_encoder(x_train_c)

                z = torch.cat((z_ex, z_mu, z_cn), 1)
                z = F.normalize(z, p=2, dim=0)
                preds = self.regressor(z)

                positive_indices, negative_indices = generate_triplets_indices(
                    y=y_train.cpu().detach().numpy(),
                    positive_range=self.positive_range,
                    negative_range=self.negative_range,
                )
                loss = self.triplet_loss(
                    z,
                    z[positive_indices],
                    z[negative_indices],
                ) + self.regression_loss(preds, y_train)

                epoch_train_loss += loss.item()

                moli_optimiser.zero_grad()
                loss.backward()
                moli_optimiser.step()
            # early stopping
            if val_loader is not None:
                with torch.no_grad():
                    self.expression_encoder.eval()
                    self.mutation_encoder.eval()
                    self.cna_encoder.eval()
                    self.regressor.eval()
                    for (x_val_e, x_val_m, x_val_c, y_val) in val_loader:
                        x_val_e = x_val_e.to(device)
                        x_val_m = x_val_m.to(device)
                        x_val_c = x_val_c.to(device)
                        y_val = y_val.to(device)

                        z_ex = self.expression_encoder(x_val_e)
                        z_mu = self.mutation_encoder(x_val_m)
                        z_cn = self.cna_encoder(x_val_c)
                        z = torch.cat((z_ex, z_mu, z_cn), 1)
                        z = F.normalize(z, p=2, dim=0)
                        preds = self.regressor(z)

                        positive_indices, negative_indices = generate_triplets_indices(
                            y=y_val.cpu().detach().numpy(),
                            positive_range=self.positive_range,
                            negative_range=self.negative_range,
                        )
                        epoch_val_loss = self.triplet_loss(
                            z,
                            z[positive_indices],
                            z[negative_indices],
                        ) + self.regression_loss(preds, y_val)
                        epoch_val_loss = epoch_val_loss.item()
                        print(f"Epoch {epoch + 1}/{self.epochs} - Train Loss: {epoch_train_loss}, "
                              f"Val Loss: {epoch_val_loss}")
                        if last_val_loss is None:
                            last_val_loss = epoch_val_loss
                        if epoch_val_loss > last_val_loss:
                            patience -= 1
                        else:
                            patience = 5
                        if patience == 0:
                            return
            else:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {epoch_train_loss}")

    def forward_with_features(self, expression, mutation, cna):
        left_out = self.expression_encoder(expression)
        middle_out = self.mutation_encoder(mutation)
        right_out = self.cna_encoder(cna)
        left_middle_right = torch.cat((left_out, middle_out, right_out), 1)
        return [self.regressor(left_middle_right), left_middle_right]

    def forward(self, expression, mutation, cna):
        if not self.model_initialized:
            self.initialize_model(expression, mutation, cna)
        left_out = self.expression_encoder(expression)
        middle_out = self.mutation_encoder(mutation)
        right_out = self.cna_encoder(cna)
        left_middle_right = torch.cat((left_out, middle_out, right_out), 1)
        return self.regressor(left_middle_right)

    def force_initialize(self, dataloader):
        """Force initialize the model by running a dummy forward pass."""
        for (x_train_e, x_train_m, x_train_c, y_train) in dataloader:
            self.forward(x_train_e, x_train_m, x_train_c)
            break

    def predict(self, gene_expression: np.ndarray, mutation: np.ndarray, cnv: np.ndarray):
        with torch.no_grad():
            self.expression_encoder.eval()
            self.mutation_encoder.eval()
            self.cna_encoder.eval()
            self.regressor.eval()
            z_ex = self.expression_encoder(torch.from_numpy(gene_expression.astype(np.float32)))
            z_mu = self.mutation_encoder(torch.from_numpy(mutation.astype(np.float32)))
            z_cn = self.cna_encoder(torch.from_numpy(cnv.astype(np.float32)))
            z = torch.cat((z_ex, z_mu, z_cn), 1)
            z = F.normalize(z, p=2, dim=0)
            preds = self.regressor(z)
        return preds.numpy()



def create_device(gpu_number):
    if torch.cuda.is_available():
        if gpu_number is None:
            free_gpu_id = get_free_gpu()
        else:
            free_gpu_id = gpu_number
        device = torch.device(f"cuda:{free_gpu_id}")
    else:
        device = torch.device("cpu")
    return device


def get_free_gpu():
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.free"]
    )
    gpu_df = pd.read_csv(BytesIO(gpu_stats), names=["memory.free"], skiprows=1)
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    return gpu_df["memory.free"].idxmax()
