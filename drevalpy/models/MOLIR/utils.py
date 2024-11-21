"""
Utility functions for the MOLIR model.

Original authors of MOLI: Sharifi-Noghabi et al. (2019, 10.1093/bioinformatics/btz318)
Code adapted from: Hauptmann et al. (2023, 10.1186/s12859-023-05166-7),
https://github.com/kramerlab/Multi-Omics_analysis
"""

import os
import random
import secrets
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from torch import nn
from torch.utils.data import DataLoader, Dataset

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


class RegressionDataset(Dataset):
    """Dataset for regression tasks for the data loader."""

    def __init__(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
    ) -> None:
        """
        Initializes the dataset by setting the output and the cell line input.

        :param output: drug response dataset
        :param cell_line_input: omics features of the cell lines
        """
        self.output = output
        self.cell_line_input = cell_line_input

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32]:
        """
        Overwrites the getitem method.

        :param idx: index of the sample
        :returns: gene expression, mutations, copy number variation, and response of the sample as numpy arrays
        """
        response: np.float32 = np.float32(self.output.response[idx])

        cell_line_id = str(self.output.cell_line_ids[idx])
        gene_expression: np.ndarray = self.cell_line_input.features[cell_line_id]["gene_expression"].astype(np.float32)
        mutations: np.ndarray = self.cell_line_input.features[cell_line_id]["mutations"].astype(np.float32)
        copy_number: np.ndarray = self.cell_line_input.features[cell_line_id]["copy_number_variation_gistic"].astype(
            np.float32
        )

        return gene_expression, mutations, copy_number, response

    def __len__(self) -> int:
        """
        Overwrites the len method.

        :returns: number of samples in the dataset
        """
        return len(self.output.response)


def generate_triplets_indices(
    y: np.ndarray,
    positive_range: float,
    negative_range: float,
    random_seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates triplets for the MOLIR model.

    The positive and negative range are determined by the standard deviation of the response values. A sample is
    considered positive if its response value is within the positive range of the label. The positive range is ±10%
    of the standard deviation of all response values. A sample is considered negative if its response value is at
    least one standard deviation away from the response value of the sample.
    :param y: response values
    :param positive_range: positive range for the triplet loss
    :param negative_range: negative range for the triplet loss
    :param random_seed: random seed for reproducibility
    :returns: positive and negative sample indices for each sample
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    positive_sample_indices = []
    negative_sample_indices = []
    # Iterate over each label in the dataset
    for idx_current_label, current_label in enumerate(y):
        positive_class_indices = _get_positive_class_indices(current_label, idx_current_label, y, positive_range)
        positive_sample_idx = np.random.choice(positive_class_indices, 1)[0]
        negative_class_indices = _get_negative_class_indices(current_label, y, negative_range)
        negative_sample_idx = np.random.choice(negative_class_indices, 1)[0]
        positive_sample_indices.append(positive_sample_idx)
        negative_sample_indices.append(negative_sample_idx)
    return np.array(positive_sample_indices), np.array(negative_sample_indices)


def _get_positive_class_indices(label: np.float32, idx_label: int, y: np.ndarray, positive_range: float) -> np.ndarray:
    """
    Find the samples that are within the positive range of the label except the label itself.

    If there is no similar sample within the positive range, the method returns the closest sample to the label.
    :param label: response of interest
    :param idx_label: index of the response of interest
    :param y: all responses
    :param positive_range: 0.1 * the standard deviation of all training responses
    :returns: indices of the samples that can be considered positive examples (=similar to the response of interest)
    """
    indices_similar_samples = np.where(np.logical_and(label - positive_range <= y, y <= label + positive_range))[0]
    indices_similar_samples = np.delete(indices_similar_samples, np.where(indices_similar_samples == idx_label))
    if len(indices_similar_samples) == 0:
        # return the closest samples to the label except the label itself
        indices_similar_samples = np.array([np.argsort(np.abs(y - label))[1]])
    return indices_similar_samples


def _get_negative_class_indices(label: np.float32, y: np.ndarray, negative_range: float) -> np.ndarray:
    """
    Finds dissimilar samples to the label.

    If there is no dissimilar sample within the negative range, the method returns the sample that is the furthest away.
    :param label: reponse of interest
    :param y: all responses
    :param negative_range: 1 * the standard deviation of all training responses
    :returns: indices of the samples that can be considered negative examples (=dissimilar to the response of interest)
    """
    dissimilar_samples = np.where(np.logical_or(label - negative_range >= y, y >= label + negative_range))[0]
    if len(dissimilar_samples) == 0:
        # return the sample that is the furthest away from the label
        dissimilar_samples = np.argsort(np.abs(y - label))[-1:]
    return dissimilar_samples


def make_ranges(output: DrugResponseDataset) -> tuple[float, float]:
    """
    Compute the positive and negative range for the triplet loss.

    :param output: drug response dataset
    :returns: positive and negative range for the triplet loss
    """
    positive_range = float(np.std(output.response) * 0.1)
    negative_range = float(np.std(output.response))
    return positive_range, negative_range


def create_dataset_and_loaders(
    batch_size: int,
    output_train: DrugResponseDataset,
    cell_line_input: FeatureDataset,
    output_earlystopping: Optional[DrugResponseDataset] = None,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Creates the RegressionDataset (torch Dataset) and the DataLoader for the training and validation data.

    :param batch_size: specified batch size
    :param output_train: response values for the training data
    :param cell_line_input: omic input features of the cell lines
    :param output_earlystopping: early stopping dataset
    :returns: training and validation data loaders
    """
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
    """
    Determines the dimensions of the omics data for the creation of the input layers.

    :param cell_line_input: omic input features of the cell lines
    :returns: dimensions of the gene expression, mutations, and copy number variation data
    """
    first_item = next(iter(cell_line_input.features.values()))
    dim_gex = first_item["gene_expression"].shape[0]
    dim_mut = first_item["mutations"].shape[0]
    dim_cnv = first_item["copy_number_variation_gistic"].shape[0]
    return dim_gex, dim_mut, dim_cnv


class MOLIEncoder(nn.Module):
    """
    Encoders of the MOLIR model, which is identical to the encoders of the original MOLI model.

    The MOLIR model has three encoders for the gene expression, mutations, and copy number variation data which are
    trained together.
    """

    def __init__(self, input_size: int, output_size: int, dropout_rate: float) -> None:
        """
        Initializes the encoder for the MOLIR model.

        :param input_size: input size determined by feature selection.
        :param output_size: output size of the encoder, set as hyperparameter.
        :param dropout_rate: dropout rate for regularization, set as hyperparameter.
        """
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        :param x: omic input features
        :returns: encoded omic features
        """
        return self.encode(x)


class MOLIRegressor(nn.Module):
    """
    Regressor of the MOLIR model.

    It is identical to the regressor of the original MOLI model, except for the omission of the final sigmoid
    activation function. After the three encoders, the encoded features are concatenated and fed into the regressor.
    """

    def __init__(self, input_size: int, dropout_rate: float) -> None:
        """
        Initializes the regressor for the MOLIR model.

        :param input_size: determined by the output sizes of the encoders.
        :param dropout_rate: set as hyperparameter.
        """
        super().__init__()
        self.regressor = nn.Sequential(nn.Linear(input_size, 1), nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the regressor.

        :param x: concatenated encoded features
        :returns: predicted drug response
        """
        return self.regressor(x)


class MOLIModel(pl.LightningModule):
    """
    PyTorch Lightning module for the MOLIR model.

    The architecture of the MOLIR model is identical to the MOLI model, except for the omission of the final sigmoid
    layer and the usage of a regression MSE loss instead of a binary cross-entropy loss. Additionally, early stopping is
    added instead of tuning the number of epochs as hyperparameter.
    """

    def __init__(
        self, hpams: dict[str, int | float], input_dim_expr: int, input_dim_mut: int, input_dim_cnv: int
    ) -> None:
        """
        Initializes the MOLIR model.

        The MOLIR model uses a combined loss function of a triplet margin loss for the concatenated representation
        and an MSE loss for the regression loss.

        :param hpams: includes mini_batch, layer dimensions (h_dim1, h_dim2, h_dim3), learning_rate, dropout_rate,
            weight decay, gamma, epochs, and margin.
        :param input_dim_expr: determined by the feature selection of the gene expression data.
        :param input_dim_mut: determined by dataset size
        :param input_dim_cnv: determined by dataset size
        """
        super().__init__()
        self.save_hyperparameters()

        self.mini_batch = int(hpams["mini_batch"])
        self.h_dim1 = int(hpams["h_dim1"])
        self.h_dim2 = int(hpams["h_dim2"])
        self.h_dim3 = int(hpams["h_dim3"])
        self.lr = hpams["learning_rate"]
        self.dropout_rate = hpams["dropout_rate"]
        self.weight_decay = hpams["weight_decay"]
        self.gamma = hpams["gamma"]
        self.epochs = int(hpams["epochs"])
        self.triplet_loss = nn.TripletMarginLoss(margin=hpams["margin"], p=2)
        self.regression_loss = nn.MSELoss()
        # Positive and Negative range for triplet loss, determined by the standard deviation of the training responses,
        # set in fit method
        self.positive_range = 1.0
        self.negative_range = 1.0
        # Checkpoint callback for early stopping, set in fit method
        self.checkpoint_callback: pl.callbacks.ModelCheckpoint | None = None

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
        """
        Trains the MOLIR model.

        First, the ranges for the triplet loss are determined using the standard deviation of the training responses.
        Then, the training and validation data loaders are created. The model is trained using the Lightning Trainer
        with an early stopping callback and patience of 5.
        :param output_train: training dataset containing the response output
        :param cell_line_input: feature dataset containing the omics data of the cell lines
        :param output_earlystopping: early stopping dataset
        :param patience: for early stopping
        """
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
            [secrets.choice("0123456789abcdef") for _ in range(20)]
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

        If there was enough training data to train the model, the model from the best epoch was saved in the checkpoint
        callback and is loaded now. If there was not enough training data, the model is only randomly initialized.
        :param gene_expression: gene expression data
        :param mutations: mutation data
        :param copy_number: copy number variation data
        :returns: predicted drug response
        """
        # load best model
        if hasattr(self, "checkpoint_callback") and self.checkpoint_callback is not None:
            best_model = MOLIModel.load_from_checkpoint(self.checkpoint_callback.best_model_path)
        else:
            best_model = self
        # convert to torch tensors
        gene_expression_tensor = torch.from_numpy(gene_expression).float().to(best_model.device)
        mutations_tensor = torch.from_numpy(mutations).float().to(best_model.device)
        copy_number_tensor = torch.from_numpy(copy_number).float().to(best_model.device)
        best_model.eval()
        with torch.no_grad():
            z = best_model._encode_and_concatenate(gene_expression_tensor, mutations_tensor, copy_number_tensor)
            preds = best_model.regressor(z)
        return preds.squeeze().cpu().detach().numpy()

    def _encode_and_concatenate(
        self, gene_expression: torch.Tensor, mutations: torch.Tensor, copy_number: torch.Tensor
    ) -> torch.Tensor:
        """
        Encodes the input modalities, concatenates, and normalizes the resulting embeddings.

        :param gene_expression: gene expression data
        :param mutations: mutation data
        :param copy_number: copy number variation data
        :returns: concatenated, normalized embeddings
        """
        z_ex = self.expression_encoder(gene_expression)
        z_mu = self.mutation_encoder(mutations)
        z_cn = self.cna_encoder(copy_number)

        z = torch.cat((z_ex, z_mu, z_cn), dim=1)
        z = nn.functional.normalize(z, p=2, dim=0)
        return z

    def forward(self, x_gene: torch.Tensor, x_mutation: torch.Tensor, x_cna: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MOLIR model.

        :param x_gene: gene expression input
        :param x_mutation: mutation input
        :param x_cna: copy number variation input
        :returns: predicted drug response
        """
        z = self._encode_and_concatenate(x_gene, x_mutation, x_cna)
        preds = self.regressor(z)
        return preds

    def _compute_loss(self, z: torch.Tensor, preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined triplet loss and regression loss.

        :param z: concatenated, normalized embeddings on which the triplet loss is calculated
        :param preds: predicted drug response on which the regression loss is calculated
        :param y: true drug response
        :returns: combined loss
        """
        positive_indices, negative_indices = generate_triplets_indices(
            y.cpu().detach().numpy(), self.positive_range, self.negative_range
        )

        triplet_loss = self.triplet_loss(z, z[positive_indices], z[negative_indices])
        regression_loss = self.regression_loss(preds.squeeze(), y)
        return triplet_loss + regression_loss

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step of the MOLIR model.

        :param batch: batch of gene expression, mutations, copy number variation, and response
        :param batch_idx: index of the batch
        :returns: combined loss
        """
        gene_expression, mutations, copy_number, response = batch

        # Encode and concatenate
        z = self._encode_and_concatenate(gene_expression, mutations, copy_number)

        # Get predictions
        preds = self.regressor(z)

        # Compute loss
        loss = self._compute_loss(z, preds, response)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step of the MOLIR model.

        :param batch: batch of gene expression, mutations, copy number variation, and response
        :param batch_idx: index of the batch
        :returns: combined loss
        """
        gene_expression, mutations, copy_number, response = batch

        # Encode and concatenate
        z = self._encode_and_concatenate(gene_expression, mutations, copy_number)

        # Get predictions
        preds = self.regressor(z)

        # Compute loss
        val_loss = self._compute_loss(z, preds, response)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Overwrites the configure_optimizers method from PyTorch Lightning.

        :returns: optimizers for the MOLIR expression, mutation, copy number variation encoders, and regressor
        """
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
