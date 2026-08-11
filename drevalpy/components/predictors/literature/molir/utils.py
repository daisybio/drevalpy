"""Utility functions for the MOLIR model.

Original authors of MOLI: Sharifi-Noghabi et al. (2019, 10.1093/bioinformatics/btz318)
Code adapted from: Hauptmann et al. (2023, 10.1186/s12859-023-05166-7),
https://github.com/kramerlab/Multi-Omics_analysis
"""

import secrets
from collections.abc import Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from torch import nn
from torch.utils.data import DataLoader
from upath import UPath as Path

from drevalpy.types.data.tensor_data import make_pair_loader
from drevalpy.utils.torch_io import load_state_dict


def generate_triplets_indices(
    y: np.ndarray,
    positive_range: float,
    negative_range: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates triplets for the MOLIR model.

    The positive and negative range are determined by the standard deviation of the response values. A sample is
    considered positive if its response value is within the positive range of the label. The positive range is ±10%
    of the standard deviation of all response values. A sample is considered negative if its response value is at
    least one standard deviation away from the response value of the sample.

    :param y: response values
    :param positive_range: positive range for the triplet loss
    :param negative_range: negative range for the triplet loss
    :param random_seed: random seed for reproducibility

    :returns: returns: positive and negative sample indices for each sample
    """
    rng = np.random.default_rng(random_seed)
    positive_sample_indices = []
    negative_sample_indices = []
    # Iterate over each label in the dataset
    for idx_current_label, current_label in enumerate(y):
        positive_class_indices = _get_positive_class_indices(current_label, idx_current_label, y, positive_range)
        positive_sample_idx = rng.choice(positive_class_indices, 1)[0]
        negative_class_indices = _get_negative_class_indices(current_label, y, negative_range)
        negative_sample_idx = rng.choice(negative_class_indices, 1)[0]
        positive_sample_indices.append(positive_sample_idx)
        negative_sample_indices.append(negative_sample_idx)
    return np.array(positive_sample_indices), np.array(negative_sample_indices)


def _get_positive_class_indices(label: np.float32, idx_label: int, y: np.ndarray, positive_range: float) -> np.ndarray:
    """Find the samples that are within the positive range of the label except the label itself.

    If there is no similar sample within the positive range, the method returns the closest sample to the label.

    :param label: response of interest
    :param idx_label: index of the response of interest
    :param y: all responses
    :param positive_range: 0.1 * the standard deviation of all training responses

    :returns: Indices of samples considered positive examples for the response of interest.
    """
    indices_similar_samples = np.where(np.logical_and(label - positive_range <= y, y <= label + positive_range))[0]
    indices_similar_samples = np.delete(indices_similar_samples, np.where(indices_similar_samples == idx_label))
    if len(indices_similar_samples) == 0 and len(y) > 1:
        # return the closest samples to the label except the label itself
        indices_similar_samples = np.array([np.argsort(np.abs(y - label))[1]])
    elif len(indices_similar_samples) == 0:
        # this happens only when there is just one sample in the validation dataset
        indices_similar_samples = np.where(y == label)[0]
    return indices_similar_samples


def _get_negative_class_indices(label: np.float32, y: np.ndarray, negative_range: float) -> np.ndarray:
    """Finds dissimilar samples to the label.

    If there is no dissimilar sample within the negative range, the method returns the sample that is the furthest away.

    :param label: reponse of interest
    :param y: all responses
    :param negative_range: 1 * the standard deviation of all training responses

    :returns: Indices of samples considered negative examples for the response of interest.
    """
    dissimilar_samples = np.where(np.logical_or(label - negative_range >= y, y >= label + negative_range))[0]
    if len(dissimilar_samples) == 0:
        # return the sample that is the furthest away from the label
        dissimilar_samples = np.argsort(np.abs(y - label))[-1:]
    return dissimilar_samples


def create_dataset_and_loaders(
    batch_size: int,
    gene_expression: np.ndarray,
    mutations: np.ndarray,
    copy_number: np.ndarray,
    response: np.ndarray,
    pair_idx: np.ndarray,
    val_gene_expression: np.ndarray | None = None,
    val_mutations: np.ndarray | None = None,
    val_copy_number: np.ndarray | None = None,
    val_response: np.ndarray | None = None,
    val_pair_idx: np.ndarray | None = None,
) -> tuple[DataLoader, DataLoader | None]:
    """Create DataLoaders using lazy pair-level index lookup.

    :param batch_size: specified batch size
    :param gene_expression: entity-level gene expression matrix (n_entities, n_features)
    :param mutations: entity-level mutation matrix (n_entities, n_features)
    :param copy_number: entity-level copy number matrix (n_entities, n_features)
    :param response: response values for training (n_pairs,)
    :param pair_idx: indices into entity matrices for each pair (n_pairs,)
    :param val_gene_expression: validation gene expression matrix (entity-level)
    :param val_mutations: validation mutation matrix (entity-level)
    :param val_copy_number: validation copy number matrix (entity-level)
    :param val_response: validation response values
    :param val_pair_idx: validation pair indices into val entity matrices

    :returns: training and validation data loaders
    :raises ValueError: If val_mutations or val_copy_number is None when validation data is provided.
    """
    resp_col = response.reshape(-1, 1) if response.ndim == 1 else response
    train_loader = make_pair_loader(
        (gene_expression, pair_idx),
        (mutations, pair_idx),
        (copy_number, pair_idx),
        response=resp_col,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    val_loader = None
    if val_gene_expression is not None and val_response is not None and val_pair_idx is not None:
        if val_mutations is None or val_copy_number is None:
            msg = "val_mutations and val_copy_number are required when val_gene_expression is provided"
            raise ValueError(msg)
        val_resp_col = val_response.reshape(-1, 1) if val_response.ndim == 1 else val_response
        val_loader = make_pair_loader(
            (val_gene_expression, val_pair_idx),
            (val_mutations, val_pair_idx),
            (val_copy_number, val_pair_idx),
            response=val_resp_col,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
    return train_loader, val_loader


def _realign_omic_matrix(
    values: np.ndarray,
    model_features: Sequence[str] | np.ndarray,
    meta_feature_names: Sequence[str] | np.ndarray,
) -> np.ndarray:
    """Align prediction-time omics columns to the feature order stored on the trained model.

    :param values: Omic feature matrix in incoming column order.
    :param model_features: Feature names used by the trained model.
    :param meta_feature_names: Feature names available in the incoming data.

    :returns: Matrix with columns reordered to match *model_features*.
    """
    if values.shape[1] == len(model_features):
        return values
    realigned = np.zeros((values.shape[0], len(model_features)))
    lookup_table = {feature: i for i, feature in enumerate(meta_feature_names)}
    for column, feature in enumerate(model_features):
        if feature in lookup_table:
            realigned[:, column] = values[:, lookup_table[feature]]
    return realigned


class MOLIEncoder(nn.Module):
    """Encoders of the MOLIR model, which is identical to the encoders of the original MOLI model.

    The MOLIR model has three encoders for the gene expression, mutations, and copy number variation data which are
    trained together.
    """

    def __init__(self, input_size: int, output_size: int, dropout_rate: float) -> None:
        """Initializes the encoder for the MOLIR model.

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
        """Forward pass of the encoder.

        :param x: omic input features

        :returns: returns: encoded omic features
        """
        return self.encode(x)


class MOLIRegressor(nn.Module):
    """Regressor of the MOLIR model.

    It is identical to the regressor of the original MOLI model, except for the omission of the final sigmoid
    activation function. After the three encoders, the encoded features are concatenated and fed into the regressor.
    """

    def __init__(self, input_size: int, dropout_rate: float) -> None:
        """Initializes the regressor for the MOLIR model.

        :param input_size: determined by the output sizes of the encoders.
        :param dropout_rate: set as hyperparameter.
        """
        super().__init__()
        self.regressor = nn.Sequential(nn.Linear(input_size, 1), nn.Dropout(dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the regressor.

        :param x: concatenated encoded features

        :returns: returns: predicted drug response
        """
        return self.regressor(x)


class MOLIModel(pl.LightningModule):
    """PyTorch Lightning module for the MOLIR model.

    The architecture of the MOLIR model is identical to the MOLI model, except for the omission of the final sigmoid
    layer and the usage of a regression MSE loss instead of a binary cross-entropy loss. Additionally, early stopping is
    added instead of tuning the number of epochs as hyperparameter.
    """

    def __init__(
        self, hpams: dict[str, int | float], input_dim_expr: int, input_dim_mut: int, input_dim_cnv: int
    ) -> None:
        """Initializes the MOLIR model.

        The MOLIR model uses a combined loss function of a triplet margin loss for the concatenated representation
        and an MSE loss for the regression loss.

        :param hpams: Hyperparameters such as ``mini_batch``, layer sizes, learning rate, and
            margin.
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
        gene_expression: np.ndarray,
        mutations: np.ndarray,
        copy_number: np.ndarray,
        response: np.ndarray,
        pair_idx: np.ndarray,
        val_gene_expression: np.ndarray | None = None,
        val_mutations: np.ndarray | None = None,
        val_copy_number: np.ndarray | None = None,
        val_response: np.ndarray | None = None,
        val_pair_idx: np.ndarray | None = None,
        patience: int = 5,
        model_checkpoint_dir: str | Path = "checkpoints",
        wandb_project: str | None = None,
    ) -> None:
        """Trains the MOLIR model.

        First, the ranges for the triplet loss are determined using the standard deviation of the training responses.
        Then, the training and validation data loaders are created. The model is trained using the Lightning Trainer
        with an early stopping callback and patience of 5.

        :param gene_expression: entity-level gene expression matrix (n_entities, n_features)
        :param mutations: entity-level mutation matrix (n_entities, n_features)
        :param copy_number: entity-level copy number matrix (n_entities, n_features)
        :param response: training response values (n_pairs,)
        :param pair_idx: indices into entity matrices for each pair (n_pairs,)
        :param val_gene_expression: validation entity-level gene expression matrix
        :param val_mutations: validation entity-level mutation matrix
        :param val_copy_number: validation entity-level copy number matrix
        :param val_response: validation response values
        :param val_pair_idx: validation pair indices into val entity matrices
        :param patience: for early stopping
        :param model_checkpoint_dir: directory to save the model checkpoints
        :param wandb_project: Optional Weights & Biases project name for Lightning logging.
        """
        std = float(np.std(response))
        self.positive_range = std * 0.1
        self.negative_range = std

        train_loader, val_loader = create_dataset_and_loaders(
            batch_size=self.mini_batch,
            gene_expression=gene_expression,
            mutations=mutations,
            copy_number=copy_number,
            response=response,
            pair_idx=pair_idx,
            val_gene_expression=val_gene_expression,
            val_mutations=val_mutations,
            val_copy_number=val_copy_number,
            val_response=val_response,
            val_pair_idx=val_pair_idx,
        )

        # Train the model
        monitor = "train_loss" if (val_loader is None) else "val_loss"

        early_stop_callback = EarlyStopping(monitor=monitor, mode="min", patience=patience)
        name = "version-" + "".join(
            [secrets.choice("0123456789abcdef") for _ in range(20)]
        )  # preventing conflicts of filenames
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=Path(model_checkpoint_dir) / name,
            monitor=monitor,
            mode="min",
            save_top_k=1,
            save_weights_only=True,
        )

        # Set up wandb logger if project is provided
        loggers = []
        if wandb_project is not None:
            from pytorch_lightning.loggers import WandbLogger

            logger = WandbLogger(project=wandb_project, log_model=False)
            loggers.append(logger)

        # Initialize the Lightning trainer
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            logger=loggers if loggers else True,  # Use default logger if no wandb
            callbacks=[
                early_stop_callback,
                self.checkpoint_callback,
                TQDMProgressBar(refresh_rate=0),
            ],
            devices=1,
            enable_model_summary=False,
        )
        if val_loader is None:
            trainer.fit(self, train_loader)
        else:
            trainer.fit(self, train_loader, val_loader)
        # load best model
        if self.checkpoint_callback.best_model_path is not None:
            checkpoint = load_state_dict(self.checkpoint_callback.best_model_path)
            self.load_state_dict(checkpoint["state_dict"])

    def predict(
        self,
        gene_expression: np.ndarray,
        mutations: np.ndarray,
        copy_number: np.ndarray,
    ) -> np.ndarray:
        """Perform prediction on given input data.

        If there was enough training data to train the model, the model from the best epoch was saved in the checkpoint
        callback and is loaded now. If there was not enough training data, the model is only randomly initialized.

        :param gene_expression: gene expression data
        :param mutations: mutation data
        :param copy_number: copy number variation data

        :returns: returns: predicted drug response
        """
        # convert to torch tensors
        gene_expression_tensor = torch.from_numpy(gene_expression).float().to(self.device)
        mutations_tensor = torch.from_numpy(mutations).float().to(self.device)
        copy_number_tensor = torch.from_numpy(copy_number).float().to(self.device)
        self.eval()
        with torch.no_grad():
            z = self._encode_and_concatenate(gene_expression_tensor, mutations_tensor, copy_number_tensor)
            preds = self.regressor(z)
        return preds.squeeze().cpu().detach().numpy()

    def _encode_and_concatenate(
        self, gene_expression: torch.Tensor, mutations: torch.Tensor, copy_number: torch.Tensor
    ) -> torch.Tensor:
        """Encodes the input modalities, concatenates, and normalizes the resulting embeddings.

        :param gene_expression: gene expression data
        :param mutations: mutation data
        :param copy_number: copy number variation data

        :returns: returns: concatenated, normalized embeddings
        """
        z_ex = self.expression_encoder(gene_expression)
        z_mu = self.mutation_encoder(mutations)
        z_cn = self.cna_encoder(copy_number)

        z = torch.cat((z_ex, z_mu, z_cn), dim=1)
        z = nn.functional.normalize(z, p=2, dim=0)
        return z

    def forward(self, x_gene: torch.Tensor, x_mutation: torch.Tensor, x_cna: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MOLIR model.

        :param x_gene: gene expression input
        :param x_mutation: mutation input
        :param x_cna: copy number variation input

        :returns: returns: predicted drug response
        """
        z = self._encode_and_concatenate(x_gene, x_mutation, x_cna)
        preds = self.regressor(z)
        return preds

    def _compute_loss(self, z: torch.Tensor, preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the combined triplet loss and regression loss.

        :param z: concatenated, normalized embeddings on which the triplet loss is calculated
        :param preds: predicted drug response on which the regression loss is calculated
        :param y: true drug response

        :returns: returns: combined loss
        """
        positive_indices, negative_indices = generate_triplets_indices(
            y.cpu().detach().numpy(), self.positive_range, self.negative_range
        )

        triplet_loss = self.triplet_loss(z, z[positive_indices], z[negative_indices])
        regression_loss = self.regression_loss(preds.squeeze(), y)
        return triplet_loss + regression_loss

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step of the MOLIR model.

        :param batch: batch of gene expression, mutations, copy number variation, and response
        :param batch_idx: index of the batch

        :returns: returns: combined loss
        """
        gene_expression, mutations, copy_number, response = batch
        response = response.squeeze(-1)

        # Encode and concatenate
        z = self._encode_and_concatenate(gene_expression, mutations, copy_number)

        # Get predictions
        preds = self.regressor(z)

        # Compute loss
        loss = self._compute_loss(z, preds, response)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step of the MOLIR model.

        :param batch: batch of gene expression, mutations, copy number variation, and response
        :param batch_idx: index of the batch

        :returns: returns: combined loss
        """
        gene_expression, mutations, copy_number, response = batch
        response = response.squeeze(-1)

        # Encode and concatenate
        z = self._encode_and_concatenate(gene_expression, mutations, copy_number)

        # Get predictions
        preds = self.regressor(z)

        # Compute loss
        val_loss = self._compute_loss(z, preds, response)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Overwrites the configure_optimizers method from PyTorch Lightning.

        :returns: returns: optimizers for the MOLIR expression, mutation, copy number variation encoders, and regressor
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
