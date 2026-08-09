"""Utility functions for the SuperFELTR model."""

import secrets

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from torch import nn
from upath import UPath as Path

from drevalpy.components.core.utils.lightning_metrics_mixin import RegressionMetricsMixin
from drevalpy.components.predictors._tensor_data import make_pair_loader
from drevalpy.components.predictors.literature.molir.utils import (
    generate_triplets_indices,
)


class SuperFELTEncoder(pl.LightningModule):
    """SuperFELT encoder definition for a single omic type, i.e., gene expression, mutation, or copy number variation.

    Very similar to MOLIEncoder, but with BatchNorm1d before ReLU.
    """

    def __init__(
        self, input_size: int, hpams: dict[str, int | float | dict], omic_type: str, ranges: tuple[float, float]
    ) -> None:
        """Initializes the SuperFELTEncoder.

        Save_hyperparameters is turned on to facilitate loading the model from a checkpoint.

        :param input_size: determined by the variance threshold feature selection
        :param hpams: hyperparameters for the model
        :param omic_type: gene expression, mutation, or copy number variation
        :param ranges: positive and negative ranges for the triplet loss

        :raises ValueError: if the hyperparameters are not of the correct type
        """
        super().__init__()
        self.save_hyperparameters()
        if (
            not isinstance(hpams["dropout_rate"], float)
            or not isinstance(hpams["margin"], float)
            or not isinstance(hpams["learning_rate"], float)
            or not isinstance(hpams["weight_decay"], float)
        ):
            raise ValueError("dropout_rate, margin, learning_rate, and weight_decay must be floats!")

        self.omic_type = omic_type
        output_size = self._get_output_size(hpams)

        # only change vs MOLIEncoder: BatchNorm1d before ReLU
        self.encode = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(hpams["dropout_rate"]),
        )
        self.lr = hpams["learning_rate"]
        self.weight_decay = hpams["weight_decay"]
        self.triplet_loss = nn.TripletMarginLoss(margin=hpams["margin"], p=2)
        self.positive_range, self.negative_range = ranges

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SuperFELTEncoder.

        :param x: input tensor

        :returns: encoded tensor
        """
        return self.encode(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Override the configure_optimizers method to use the Adam optimizer.

        :returns: Adam optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _get_output_size(self, hpams: dict[str, int | float | dict]) -> int:
        """Get the output size of the encoder based on the omic type from the hyperparameters.

        :param hpams: hyperparameters for the model

        :returns: output size of the encoder

        :raises ValueError: if the output sizes are not of the correct type
        """
        if (
            not isinstance(hpams["out_dim_expr_encoder"], int)
            or not isinstance(hpams["out_dim_mutation_encoder"], int)
            or not isinstance(hpams["out_dim_cnv_encoder"], int)
        ):
            raise ValueError("out_dim_expr_encoder, out_dim_mutation_encoder, and out_dim_cnv_encoder must be ints!")

        output_sizes = {
            "expression": hpams["out_dim_expr_encoder"],
            "mutation": hpams["out_dim_mutation_encoder"],
            "copy_number_variation_gistic": hpams["out_dim_cnv_encoder"],
        }
        output_size = output_sizes[self.omic_type]
        return output_size

    def _get_omic_data(self, data_expr: torch.Tensor, data_mut: torch.Tensor, data_cnv: torch.Tensor) -> torch.Tensor:
        """Get the omic data based on the omic type.

        :param data_expr: expression data
        :param data_mut: mutation data
        :param data_cnv: copy number variation data

        :returns: the omic data

        :raises ValueError: if the omic type is not recognized
        """
        if self.omic_type == "expression":
            data = data_expr
        elif self.omic_type == "mutation":
            data = data_mut
        elif self.omic_type == "copy_number_variation_gistic":
            data = data_cnv
        else:
            raise ValueError(f"omic_type {self.omic_type} not recognized.")
        return data

    def _compute_loss(self, encoded: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        """Computes the triplet loss.

        :param encoded: encoded data
        :param response: response data

        :returns: triplet loss
        """
        positive_indices, negative_indices = generate_triplets_indices(
            response.cpu().detach().numpy(), self.positive_range, self.negative_range
        )
        triplet_loss = self.triplet_loss(encoded, encoded[positive_indices], encoded[negative_indices])
        return triplet_loss

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Override the training_step method to compute the triplet loss.

        :param batch: batch containing the omic data and response
        :param batch_idx: index of the batch

        :returns: triplet loss
        """
        data_expr, data_mut, data_cnv, response = batch
        response = response.squeeze(-1)
        data = self._get_omic_data(data_expr, data_mut, data_cnv)
        encoded = self.encode(data)
        triplet_loss = self._compute_loss(encoded, response)
        self.log("train_loss", triplet_loss, on_step=False, on_epoch=True, prog_bar=True)
        return triplet_loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Override the validation_step method to compute the triplet loss.

        :param batch: batch containing the omic data and response
        :param batch_idx: index of the batch

        :returns: triplet loss
        """
        data_expr, data_mut, data_cnv, response = batch
        response = response.squeeze(-1)
        data = self._get_omic_data(data_expr, data_mut, data_cnv)
        encoded = self.encode(data)
        triplet_loss = self._compute_loss(encoded, response)
        self.log("val_loss", triplet_loss, on_step=False, on_epoch=True, prog_bar=True)
        return triplet_loss


class SuperFELTRegressor(RegressionMetricsMixin, pl.LightningModule):
    """SuperFELT regressor definition.

    Very similar to SuperFELT classifier, but with a regression loss and without the last sigmoid layer.
    """

    def __init__(
        self,
        input_size: int,
        hpams: dict[str, int | float | dict],
        encoders: tuple[SuperFELTEncoder, SuperFELTEncoder, SuperFELTEncoder],
    ) -> None:
        """Initializes the SuperFELTRegressor.

        The encoders are put in eval mode because they were fitted before.

        :param input_size: depends on the output of the encoders
        :param hpams: hyperparameters for the model
        :param encoders: the fitted encoders for the gene expression, mutation, and copy number variation data

        :raises ValueError: if the hyperparameters are not of the correct type
        """
        super().__init__()
        if (
            not isinstance(hpams["learning_rate"], float)
            or not isinstance(hpams["weight_decay"], float)
            or not isinstance(hpams["dropout_rate"], float)
        ):
            raise ValueError("learning_rate, weight_decay and dropout_rate must be floats!")

        self.regressor = nn.Sequential(nn.Linear(input_size, 1), nn.Dropout(hpams["dropout_rate"]))
        self.lr = float(hpams["learning_rate"])
        self.weight_decay = float(hpams["weight_decay"])
        self.encoders = encoders
        # put the encoders in eval mode
        for encoder in self.encoders:
            encoder.eval()
        self.regression_loss = nn.MSELoss()

        # Initialize metrics storage for epoch-end R^2 and PCC computation
        self._init_metrics_storage()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SuperFELTRegressor.

        :param x: input tensor

        :returns: predicted response
        """
        return self.regressor(x)

    def predict(self, data_expr: np.ndarray, data_mut: np.ndarray, data_cnv: np.ndarray) -> np.ndarray:
        """Predicts the response for the given input.

        :param data_expr: expression data
        :param data_mut: mutation data
        :param data_cnv: copy number variation data

        :returns: predicted response
        """
        data_expr_tensor, data_mut_tensor, data_cnv_tensor = map(
            lambda data: torch.from_numpy(data).float().to(self.device), [data_expr, data_mut, data_cnv]
        )
        self.eval()
        with torch.no_grad():
            encoded = self._encode_and_concatenate(data_expr_tensor, data_mut_tensor, data_cnv_tensor)
            preds = self.regressor(encoded)
        return preds.squeeze().cpu().detach().numpy()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Override the configure_optimizers method to use the Adagrad optimizer.

        :returns: Adagrad optimizer
        """
        return torch.optim.Adagrad(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _encode_and_concatenate(
        self, data_expr: torch.Tensor, data_mut: torch.Tensor, data_cnv: torch.Tensor
    ) -> torch.Tensor:
        """Encodes the omic data and concatenates the encoded tensors.

        :param data_expr: expression data
        :param data_mut: mutation data
        :param data_cnv: copy number variation data

        :returns: concatenated encoded tensor
        """
        encoded_expr = self.encoders[0].encode(data_expr)
        encoded_mut = self.encoders[1].encode(data_mut)
        encoded_cnv = self.encoders[2].encode(data_cnv)
        return torch.cat((encoded_expr, encoded_mut, encoded_cnv), dim=1)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Override the training_step method to compute the regression loss.

        :param batch: batch containing the omic data and response
        :param batch_idx: index of the batch

        :returns: regression loss
        """
        data_expr, data_mut, data_cnv, response = batch
        response = response.squeeze(-1)
        encoded = self._encode_and_concatenate(data_expr, data_mut, data_cnv)
        pred = self.regressor(encoded)
        loss = self.regression_loss(pred.squeeze(), response)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Store predictions and targets for epoch-end metrics via mixin
        self._store_predictions(pred.squeeze(), response, is_training=True)

        return loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Override the validation_step method to compute the regression loss.

        :param batch: batch containing the omic data and response
        :param batch_idx: index of the batch

        :returns: regression loss
        """
        data_expr, data_mut, data_cnv, response = batch
        response = response.squeeze(-1)
        encoded = self._encode_and_concatenate(data_expr, data_mut, data_cnv)
        pred = self.regressor(encoded)
        loss = self.regression_loss(pred.squeeze(), response)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Store predictions and targets for epoch-end metrics via mixin
        self._store_predictions(pred.squeeze(), response, is_training=False)

        return loss


def train_superfeltr_model(
    model: SuperFELTEncoder | SuperFELTRegressor,
    hpams: dict[str, int | float | dict],
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
    model_checkpoint_dir: str | Path = "superfeltr_checkpoints",
    wandb_project: str | None = None,
) -> pl.callbacks.ModelCheckpoint:
    """Trains one encoder or the regressor using lazy pair-level lookup.

    :param model: either one of the encoders or the regressor
    :param hpams: hyperparameters for the model
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
    :param patience: for early stopping, defaults to 5
    :param model_checkpoint_dir: directory to save the model checkpoints
    :param wandb_project: Optional Weights & Biases project name for Lightning logging.

    :returns: checkpoint callback with the best model

    :raises ValueError: if the epochs and mini_batch are not integers
    """
    if not isinstance(hpams["epochs"], int) or not isinstance(hpams["mini_batch"], int):
        raise ValueError("epochs and mini_batch must be integers!")

    batch_size = hpams["mini_batch"]
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

    monitor = "train_loss" if (val_loader is None) else "val_loss"
    early_stop_callback = EarlyStopping(monitor=monitor, mode="min", patience=patience)
    name = "version-" + "".join(
        [secrets.choice("0123456789abcdef") for _ in range(20)]
    )  # preventing conflicts of filenames
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(model_checkpoint_dir) / name,
        monitor=monitor,
        mode="min",
        save_top_k=1,
    )
    # Set up wandb logger if project is provided
    loggers = []
    if wandb_project is not None:
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(project=wandb_project, log_model=False)
        loggers.append(logger)

    # Initialize the Lightning trainer
    trainer = pl.Trainer(
        max_epochs=hpams["epochs"],
        logger=loggers if loggers else True,  # Use default logger if no wandb
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            TQDMProgressBar(refresh_rate=0),
        ],
    )
    if val_loader is None:
        trainer.fit(model, train_loader)
    else:
        trainer.fit(model, train_loader, val_loader)
    return checkpoint_callback
