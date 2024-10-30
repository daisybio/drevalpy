"""
Code for the MOLI model.
Original authors: Sharifi-Noghabi et al. (2019, 10.1093/bioinformatics/btz318)
Code adapted from: Hauptmann et al. (2023, 10.1186/s12859-023-05166-7),
https://github.com/kramerlab/Multi-Omics_analysis
"""

import os
import random
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from torch import nn
from torch.utils.data import DataLoader, Dataset

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


class RegressionDataset(Dataset):
    """
    Dataset for regression tasks for the data loader.
    """

    def __init__(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset = None,
    ) -> None:
        self.output = output
        self.cell_line_input = cell_line_input

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        response = self.output.response[idx].astype(np.float32)

        cell_line_id = str(self.output.cell_line_ids[idx])
        gene_expression = self.cell_line_input.features[cell_line_id]["gene_expression"].astype(np.float32)
        mutations = self.cell_line_input.features[cell_line_id]["mutations"].astype(np.float32)
        copy_number = self.cell_line_input.features[cell_line_id]["copy_number_variation_gistic"].astype(np.float32)

        return gene_expression, mutations, copy_number, response

    def __len__(self) -> int:
        """
        Overwrites the len method.
        """
        return len(self.output.response)


def generate_triplets_indices(
    y: np.ndarray,
    positive_range: float,
    negative_range: float,
    random_seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
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
        positive_class_indices = get_positive_class_indices(current_label, idx_current_label, y, positive_range)
        positive_sample_idx = np.random.choice(positive_class_indices, 1)[0]
        negative_class_indices = get_negative_class_indices(current_label, y, negative_range)
        negative_sample_idx = np.random.choice(negative_class_indices, 1)[0]
        positive_sample_indices.append(positive_sample_idx)
        negative_sample_indices.append(negative_sample_idx)
    return np.array(positive_sample_indices), np.array(negative_sample_indices)


def get_positive_class_indices(label: float, idx_label: int, y: np.ndarray, positive_range: float) -> np.ndarray:
    # find the samples that are within the positive range of the label except the label itself
    indices_similar_samples = np.where(np.logical_and(label - positive_range <= y, y <= label + positive_range))[0]
    indices_similar_samples = np.delete(indices_similar_samples, np.where(indices_similar_samples == idx_label))
    if len(indices_similar_samples) == 0:
        # return the closest samples to the label except the label itself
        indices_similar_samples = np.array([np.argsort(np.abs(y - label))[1]])
    return indices_similar_samples


def get_negative_class_indices(label: float, y: np.ndarray, negative_range: float) -> np.ndarray:
    dissimilar_samples = np.where(np.logical_or(label - negative_range >= y, y >= label + negative_range))[0]
    if len(dissimilar_samples) == 0:
        # return the sample that is the furthest away from the label
        dissimilar_samples = np.argsort(np.abs(y - label))[-1:]
    return dissimilar_samples


def make_ranges(output: DrugResponseDataset) -> tuple[float, float]:
    """
    Compute the positive and negative range for the triplet loss.
    """
    positive_range = np.std(output.response) * 0.1
    negative_range = np.std(output.response)
    return positive_range, negative_range


def create_dataset_and_loaders(
    batch_size: int,
    output_train: DrugResponseDataset,
    cell_line_input: FeatureDataset,
    output_earlystopping: Optional[DrugResponseDataset] = None,
) -> tuple[DataLoader, Optional[DataLoader]]:
    # Create datasets and dataloaders
    train_dataset = RegressionDataset(output_train, cell_line_input)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        drop_last=True,  # avoids batch norm errors if last batch < batch_size
    )

    val_loader = None
    if output_earlystopping is not None:
        val_dataset = RegressionDataset(
            output=output_earlystopping,
            cell_line_input=cell_line_input,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )
    return train_loader, val_loader


def get_dimensions_of_omics_data(cell_line_input: FeatureDataset) -> tuple[int, int, int]:
    first_item = next(iter(cell_line_input.features.values()))
    dim_gex = first_item["gene_expression"].shape[0]
    dim_mut = first_item["mutations"].shape[0]
    dim_cnv = first_item["copy_number_variation_gistic"].shape[0]
    return dim_gex, dim_mut, dim_cnv


class MOLIEncoder(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout_rate: float) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class MOLIRegressor(nn.Module):
    def __init__(self, input_size: int, dropout_rate: int) -> None:
        super().__init__()
        self.regressor = nn.Sequential(nn.Linear(input_size, 1), nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)


class MOLIModel(pl.LightningModule):
    def __init__(
        self, hpams: dict[str, Union[int, float]], input_dim_expr: int, input_dim_mut: int, input_dim_cnv: int
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.mini_batch = hpams["mini_batch"]
        self.h_dim1 = hpams["h_dim1"]
        self.h_dim2 = hpams["h_dim2"]
        self.h_dim3 = hpams["h_dim3"]
        self.lr = hpams["learning_rate"]
        self.dropout_rate = hpams["dropout_rate"]
        self.weight_decay = hpams["weight_decay"]
        self.gamma = hpams["gamma"]
        self.epochs = hpams["epochs"]
        self.triplet_loss = nn.TripletMarginLoss(margin=hpams["margin"], p=2)
        self.regression_loss = nn.MSELoss()
        # Positive and Negative range for triplet loss
        self.positive_range = None
        self.negative_range = None
        self.checkpoint_callback = None

        self.expression_encoder = MOLIEncoder(input_dim_expr, self.h_dim1, self.dropout_rate)
        self.mutation_encoder = MOLIEncoder(input_dim_mut, self.h_dim2, self.dropout_rate)
        self.cna_encoder = MOLIEncoder(input_dim_cnv, self.h_dim3, self.dropout_rate)
        self.regressor = MOLIRegressor(self.h_dim1 + self.h_dim2 + self.h_dim3, self.dropout_rate)

    def fit(
        self,
        output_train: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        output_earlystopping: Optional[DrugResponseDataset] = None,
        patience: int = 5,
    ) -> None:
        self.positive_range, self.negative_range = make_ranges(output_train)

        train_loader, val_loader = create_dataset_and_loaders(
            batch_size=self.mini_batch,
            output_train=output_train,
            cell_line_input=cell_line_input,
            output_earlystopping=output_earlystopping,
        )

        # Train the model
        monitor = "train_loss" if (val_loader is None) else "val_loss"

        early_stop_callback = EarlyStopping(monitor=monitor, mode="min", patience=patience)
        name = "version-" + "".join(
            [random.choice("0123456789abcdef") for _ in range(20)]
        )  # preventing conflicts of filenames
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=None,
            monitor=monitor,
            mode="min",
            save_top_k=1,
            filename=name,
        )

        # Initialize the Lightning trainer
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            callbacks=[
                early_stop_callback,
                self.checkpoint_callback,
                TQDMProgressBar(),
            ],
            default_root_dir=os.path.join(os.getcwd(), "moli_checkpoints/lightning_logs/" + name),
        )
        if val_loader is None:
            trainer.fit(self, train_loader)
        else:
            trainer.fit(self, train_loader, val_loader)

    def predict(
        self,
        gene_expression: np.ndarray,
        mutations: np.ndarray,
        copy_number: np.ndarray,
    ) -> np.ndarray:
        """
        Perform prediction on given input data.
        """
        # load best model
        if self.checkpoint_callback.best_model_path:
            best_model = MOLIModel.load_from_checkpoint(self.checkpoint_callback.best_model_path)
        else:
            best_model = self
        # convert to torch tensors
        gene_expression = torch.from_numpy(gene_expression).float().to(best_model.device)
        mutations = torch.from_numpy(mutations).float().to(best_model.device)
        copy_number = torch.from_numpy(copy_number).float().to(best_model.device)
        best_model.eval()
        with torch.no_grad():
            z = best_model.encode_and_concatenate(gene_expression, mutations, copy_number)
            preds = best_model.regressor(z)
        return preds.squeeze().cpu().detach().numpy()

    def encode_and_concatenate(
        self, gene_expression: torch.Tensor, mutations: torch.Tensor, copy_number: torch.Tensor
    ) -> torch.Tensor:
        """
        Encodes the input modalities (gene expression, mutations, and copy number)
        and concatenates the resulting embeddings.
        """
        z_ex = self.expression_encoder(gene_expression)
        z_mu = self.mutation_encoder(mutations)
        z_cn = self.cna_encoder(copy_number)

        z = torch.cat((z_ex, z_mu, z_cn), dim=1)
        z = nn.functional.normalize(z, p=2, dim=0)
        return z

    def forward(self, x_gene: torch.Tensor, x_mutation: torch.Tensor, x_cna: torch.Tensor) -> torch.Tensor:
        z = self.encode_and_concatenate(x_gene, x_mutation, x_cna)
        preds = self.regressor(z)
        return preds

    def compute_loss(self, z: torch.Tensor, preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined triplet loss and regression loss.
        """
        positive_indices, negative_indices = generate_triplets_indices(
            y.cpu().detach().numpy(), self.positive_range, self.negative_range
        )

        triplet_loss = self.triplet_loss(z, z[positive_indices], z[negative_indices])
        regression_loss = self.regression_loss(preds.squeeze(), y)
        return triplet_loss + regression_loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        gene_expression, mutations, copy_number, response = batch

        # Encode and concatenate
        z = self.encode_and_concatenate(gene_expression, mutations, copy_number)

        # Get predictions
        preds = self.regressor(z)

        # Compute loss
        loss = self.compute_loss(z, preds, response)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        gene_expression, mutations, copy_number, response = batch

        # Encode and concatenate
        z = self.encode_and_concatenate(gene_expression, mutations, copy_number)

        # Get predictions
        preds = self.regressor(z)

        # Compute loss
        val_loss = self.compute_loss(z, preds, response)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adagrad(
            [
                {"params": self.expression_encoder.parameters(), "lr": self.lr},
                {"params": self.mutation_encoder.parameters(), "lr": self.lr},
                {"params": self.cna_encoder.parameters(), "lr": self.lr},
                {"params": self.regressor.parameters(), "lr": self.lr},
            ],
            weight_decay=self.weight_decay,
        )
        return optimizer
