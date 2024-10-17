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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
import random
import numpy as np
from itertools import combinations


def generate_triplets_multi_features(
    x_gene_expression: np.ndarray,
    x_mutations: np.ndarray,
    x_copy_number_variation_gistic: np.ndarray,
    y: np.ndarray,
    positive_range: float,
    negative_range: float,
    num_positive_pairs: int = 10,
    num_negative_pairs: int = 10,
    random_seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Generates triplets of Anchor, Positive, and Negative samples for multiple feature types:
    gene expression, mutations, and copy number variation (GISTIC).

    Parameters:
    -----------
    x_gene_expression : np.ndarray
        Gene expression feature matrix of shape (n_samples, n_features).
    x_mutations : np.ndarray
        Mutation feature matrix of shape (n_samples, n_features).
    x_copy_number_variation_gistic : np.ndarray
        Copy number variation (GISTIC) feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Labels corresponding to each sample in the feature matrices.
    positive_range : float
        Tolerance range for identifying positive pairs (samples of the same class).
    negative_range : float
        Separation range for identifying negative pairs (samples of different classes).
    num_positive_pairs : int, optional (default=10)
        Number of Anchor-Positive pairs to generate per class.
    num_negative_pairs : int, optional (default=10)
        Number of Anchor-Negative pairs to generate per class.
    random_seed : Optional[int], optional (default=None)
        Random seed for reproducibility of the generated triplets.

    Returns:
    --------
    Dict[str, np.ndarray]
        A dictionary containing the triplets in the format [Anchor, Positive, Negative]
        for each feature type (x_gene_expression, x_mutations, x_copy_number_variation_gistic).

    Raises:
    -------
    ValueError
        If input arrays do not have matching dimensions, or if there are insufficient
        samples to create valid triplets under the given conditions.
    """

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Validate input dimensions
    if not (x_gene_expression.shape[0] == x_mutations.shape[0] == x_copy_number_variation_gistic.shape[0] == y.shape[0]):
        raise ValueError("All feature matrices and y must have the same number of samples.")

    triplets_gene_expression: List[List[np.ndarray]] = []
    triplets_mutations: List[List[np.ndarray]] = []
    triplets_copy_number_variation_gistic: List[List[np.ndarray]] = []

    # Iterate over each label in the dataset
    for current_label in y:
        positive_class_indices = get_positive_class_indices(current_label, y, positive_range)
        negative_class_indices = get_negative_class_indices(current_label, y, negative_range)

        anchor_positive_pairs = generate_anchor_positive_pairs(positive_class_indices, num_positive_pairs)
        negative_samples = generate_negative_samples(negative_class_indices, num_negative_pairs)

        # Generate triplets for each feature type (x_gene_expression, x_mutations, x_copy_number_variation_gistic)
        for anchor_positive_pair in anchor_positive_pairs:
            a_idx, p_idx = anchor_positive_pair

            # Anchor-Positive pairs for each feature type
            gene_expr_anchor, gene_expr_positive = (
                x_gene_expression[a_idx],
                x_gene_expression[p_idx],
            )
            mutations_anchor, mutations_positive = (
                x_mutations[a_idx],
                x_mutations[p_idx],
            )
            cnv_gistic_anchor, cnv_gistic_positive = (
                x_copy_number_variation_gistic[a_idx],
                x_copy_number_variation_gistic[p_idx],
            )

            # Negative samples for each feature type
            for n_idx in negative_samples:
                gene_expr_negative = x_gene_expression[n_idx]
                mutations_negative = x_mutations[n_idx]
                cnv_gistic_negative = x_copy_number_variation_gistic[n_idx]

                triplets_gene_expression.append([gene_expr_anchor, gene_expr_positive, gene_expr_negative])
                triplets_mutations.append([mutations_anchor, mutations_positive, mutations_negative])
                triplets_copy_number_variation_gistic.append([cnv_gistic_anchor, cnv_gistic_positive, cnv_gistic_negative])

    return {
        "x_gene_expression": np.array(triplets_gene_expression),
        "x_mutations": np.array(triplets_mutations),
        "x_copy_number_variation_gistic": np.array(triplets_copy_number_variation_gistic),
    }


# The helper functions remain the same:
def get_positive_class_indices(label: float, y: np.ndarray, positive_range: float) -> np.ndarray:
    return np.where(np.logical_and(label - positive_range <= y, y <= label + positive_range))[0]


def get_negative_class_indices(label: float, y: np.ndarray, negative_range: float) -> np.ndarray:
    return np.where(np.logical_or(label - negative_range >= y, y >= label + negative_range))[0]


def generate_anchor_positive_pairs(positive_class_indices: np.ndarray, num_pairs: int) -> List[Tuple[int, int]]:
    if len(positive_class_indices) <= 1:
        raise ValueError("Not enough positive samples to generate pairs.")
    if len(positive_class_indices) <= 15:
        return list(combinations(positive_class_indices, 2))
    else:
        sampled_indices = random.sample(list(positive_class_indices), k=15)
        return random.sample(list(combinations(sampled_indices, 2)), k=num_pairs)


def generate_negative_samples(negative_class_indices: np.ndarray, num_samples: int) -> List[int]:
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
        return (torch.tensor(x_e_triplet, dtype=torch.float32),
                torch.tensor(x_m_triplet, dtype=torch.float32),
                torch.tensor(x_c_triplet, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))




class MOLIClassifier(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(MOLIClassifier, self).__init__()
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
        self.lr_clf = hpams["lr_cl"]
        self.dropout_rate = hpams["dropout_rate"]
        self.weight_decay = hpams["weight_decay"]
        self.gamma = hpams["gamma"]
        self.epochs = hpams["epochs"]
        self.triplett_loss = torch.nn.TripletMarginLoss(margin=hpams["margin"], p=2)
        self.regression_loss = nn.MSELoss()
        self.model_initialized = False
        self.expression_encoder = None
        self.mutation_encoder = None
        self.cna_encoder = None
        self.classifier = None

    def initialize_model(self, x_train_e, x_train_m, x_train_c):
        _, ie_dim = x_train_e.shape
        _, im_dim = x_train_m.shape
        _, ic_dim = x_train_c.shape
        self.expression_encoder = MOLIEncoder(ie_dim, self.h_dim1, self.dropout_rate)
        self.mutation_encoder = MOLIEncoder(im_dim, self.h_dim2, self.dropout_rate)
        self.cna_encoder = MOLIEncoder(ic_dim, self.h_dim3, self.dropout_rate)
        self.classifier = MOLIClassifier(ie_dim + im_dim + ic_dim, self.dropout_rate)
        self.model_initialized = True

    def fit(
        self,
        output_train: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset = None,
        cell_line_views: List[str] = None,
        drug_views: List[str] = None,
        output_earlystopping: Optional[DrugResponseDataset] = None,
        patience: int = 5,
    ):
        device = create_device(gpu_number=None)
        train_dataset = TripletDataset(
            output=output_train,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            cell_line_views=cell_line_views,
            drug_views=drug_views,
        )
        # Assuming the triplets dictionary and labels are already generated
        # For example:
        triplets = generate_triplets_multi_features(x_gene_expression, x_mutations, x_copy_number_variation_gistic, y, positive_range, negative_range)
        labels = np.ones(triplets["x_gene_expression"].shape[0])  # Example labels, you would use the correct ones here

        # Create the dataset
        train_dataset = TripletDataset(triplets, labels)

        # Create the DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        val_loader = None
        if output_earlystopping is not None:
            val_dataset = RegressionDataset(
                output=output_earlystopping,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
                cell_line_views=cell_line_views,
                drug_views=drug_views,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.mini_batch,
                shuffle=False,
                num_workers=1,
                persistent_workers=True,
            )
        moli_optimiser = torch.optim.Adagrad(
            [
                {"params": self.expression_encoder.parameters(), "lr": self.lr_e},
                {"params": self.mutation_encoder.parameters(), "lr": self.lr_m},
                {"params": self.cna_encoder.parameters(), "lr": self.lr_c},
                {"params": self.classifier.parameters(), "lr": self.lr_clf},
            ],
            weight_decay=self.weight_decay,
        )
        cpu = device == torch.device("cpu")

        for _ in range(self.epochs):
            for batch in train_loader:
                x_train_e, x_train_m, x_train_c, y_train = batch
                x_train_e = x_train_e.to(device)
                x_train_m = x_train_m.to(device)
                x_train_c = x_train_c.to(device)
                y_train = y_train.to(device)
                self.expression_encoder = self.expression_encoder.to(device)
                self.expression_encoder.train()
                self.mutation_encoder = self.mutation_encoder.to(device)
                self.mutation_encoder.train()
                self.cna_encoder = self.cna_encoder.to(device)
                self.cna_encoder.train()
                self.classifier = self.classifier.to(device)
                self.classifier.train()

                z_ex = self.expression_encoder(x_train_e)
                z_mu = self.mutation_encoder(x_train_m)
                z_cn = self.cna_encoder(x_train_c)

                z = torch.cat((z_ex, z_mu, z_cn), 1)
                z = F.normalize(z, p=2, dim=0)
                preds = self.classifier(z)
                loss = self.loss(preds, y_train)
                moli_optimiser.zero_grad()
                loss.backward()
                moli_optimiser.step()
                # early stopping
                if val_loader is not None:
                    with torch.no_grad():
                        self.expression_encoder.eval()
                        self.mutation_encoder.eval()
                        self.cna_encoder.eval()
                        self.classifier.eval()

    def forward_with_features(self, expression, mutation, cna):
        left_out = self.expression_encoder(expression)
        middle_out = self.mutation_encoder(mutation)
        right_out = self.cna_encoder(cna)
        left_middle_right = torch.cat((left_out, middle_out, right_out), 1)
        return [self.classifier(left_middle_right), left_middle_right]

    def forward(self, expression, mutation, cna):
        if not self.model_initialized:
            self.initialize_model(expression, mutation, cna)
        left_out = self.expression_encoder(expression)
        middle_out = self.mutation_encoder(mutation)
        right_out = self.cna_encoder(cna)
        left_middle_right = torch.cat((left_out, middle_out, right_out), 1)
        return self.classifier(left_middle_right)


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
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.free"])
    gpu_df = pd.read_csv(BytesIO(gpu_stats), names=["memory.free"], skiprows=1)
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    return gpu_df["memory.free"].idxmax()
