"""Utility functions for the SuperFELTR model."""

import os
import secrets

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from torch import nn

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..MOLIR.utils import create_dataset_and_loaders, generate_triplets_indices


class SuperFELTEncoder(pl.LightningModule):
    """
    SuperFELT encoder definition for a single omic type, i.e., gene expression, mutation, or copy number variation.

    Very similar to MOLIEncoder, but with BatchNorm1d before ReLU.
    """

    def __init__(
        self, input_size: int, hpams: dict[str, int | float | dict], omic_type: str, ranges: tuple[float, float]
    ) -> None:
        """
        Initializes the SuperFELTEncoder.

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
        """
        Forward pass of the SuperFELTEncoder.

        :param x: input tensor
        :returns: encoded tensor
        """
        return self.encode(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Override the configure_optimizers method to use the Adam optimizer.

        :returns: Adam optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _get_output_size(self, hpams: dict[str, int | float | dict]) -> int:
        """
        Get the output size of the encoder based on the omic type from the hyperparameters.

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
        """
        Get the omic data based on the omic type.

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
        """
        Computes the triplet loss.

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
        """
        Override the training_step method to compute the triplet loss.

        :param batch: batch containing the omic data and response
        :param batch_idx: index of the batch
        :returns: triplet loss
        """
        data_expr, data_mut, data_cnv, response = batch
        data = self._get_omic_data(data_expr, data_mut, data_cnv)
        encoded = self.encode(data)
        triplet_loss = self._compute_loss(encoded, response)
        self.log("train_loss", triplet_loss, on_step=False, on_epoch=True, prog_bar=True)
        return triplet_loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Override the validation_step method to compute the triplet loss.

        :param batch: batch containing the omic data and response
        :param batch_idx: index of the batch
        :returns: triplet loss
        """
        data_expr, data_mut, data_cnv, response = batch
        data = self._get_omic_data(data_expr, data_mut, data_cnv)
        encoded = self.encode(data)
        triplet_loss = self._compute_loss(encoded, response)
        self.log("val_loss", triplet_loss, on_step=False, on_epoch=True, prog_bar=True)
        return triplet_loss


class SuperFELTRegressor(pl.LightningModule):
    """
    SuperFELT regressor definition.

    Very similar to SuperFELT classifier, but with a regression loss and without the last sigmoid layer.
    """

    def __init__(
        self,
        input_size: int,
        hpams: dict[str, int | float | dict],
        encoders: tuple[SuperFELTEncoder, SuperFELTEncoder, SuperFELTEncoder],
    ) -> None:
        """
        Initializes the SuperFELTRegressor.

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SuperFELTRegressor.

        :param x: input tensor
        :returns: predicted response
        """
        return self.regressor(x)

    def predict(self, data_expr: np.ndarray, data_mut: np.ndarray, data_cnv: np.ndarray) -> np.ndarray:
        """
        Predicts the response for the given input.

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
        """
        Override the configure_optimizers method to use the Adagrad optimizer.

        :returns: Adagrad optimizer
        """
        return torch.optim.Adagrad(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _encode_and_concatenate(
        self, data_expr: torch.Tensor, data_mut: torch.Tensor, data_cnv: torch.Tensor
    ) -> torch.Tensor:
        """
        Encodes the omic data and concatenates the encoded tensors.

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
        """
        Override the training_step method to compute the regression loss.

        :param batch: batch containing the omic data and response
        :param batch_idx: index of the batch
        :returns: regression loss
        """
        data_expr, data_mut, data_cnv, response = batch
        encoded = self._encode_and_concatenate(data_expr, data_mut, data_cnv)
        pred = self.regressor(encoded)
        loss = self.regression_loss(pred.squeeze(), response)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Override the validation_step method to compute the regression loss.

        :param batch: batch containing the omic data and response
        :param batch_idx: index of the batch
        :returns: regression loss
        """
        data_expr, data_mut, data_cnv, response = batch
        encoded = self._encode_and_concatenate(data_expr, data_mut, data_cnv)
        pred = self.regressor(encoded)
        loss = self.regression_loss(pred.squeeze(), response)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


def train_superfeltr_model(
    model: SuperFELTEncoder | SuperFELTRegressor,
    hpams: dict[str, int | float | dict],
    output_train: DrugResponseDataset,
    cell_line_input: FeatureDataset,
    output_earlystopping: DrugResponseDataset | None = None,
    patience: int = 5,
) -> pl.callbacks.ModelCheckpoint:
    """
    Trains one encoder or the regressor.

    First, the dataset and loaders are created. Then, the model is trained with the Lightning trainer.
    :param model: either one of the encoders or the regressor
    :param hpams: hyperparameters for the model
    :param output_train: response data for training
    :param cell_line_input: cell line omics features
    :param output_earlystopping: response data for early stopping
    :param patience: for early stopping, defaults to 5
    :returns: checkpoint callback with the best model
    :raises ValueError: if the epochs and mini_batch are not integers
    """
    if not isinstance(hpams["epochs"], int) or not isinstance(hpams["mini_batch"], int):
        raise ValueError("epochs and mini_batch must be integers!")

    train_loader, val_loader = create_dataset_and_loaders(
        batch_size=hpams["mini_batch"],
        output_train=output_train,
        cell_line_input=cell_line_input,
        output_earlystopping=output_earlystopping,
    )
    monitor = "train_loss" if (val_loader is None) else "val_loss"
    early_stop_callback = EarlyStopping(monitor=monitor, mode="min", patience=patience)
    name = "version-" + "".join(
        [secrets.choice("0123456789abcdef") for _ in range(20)]
    )  # preventing conflicts of filenames
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=None,
        monitor=monitor,
        mode="min",
        save_top_k=1,
        filename=name,
    )
    # Initialize the Lightning trainer
    trainer = pl.Trainer(
        max_epochs=hpams["epochs"],
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            TQDMProgressBar(),
        ],
        default_root_dir=os.path.join(os.getcwd(), "superfeltr_checkpoints/lightning_logs/" + name),
    )
    if val_loader is None:
        trainer.fit(model, train_loader)
    else:
        trainer.fit(model, train_loader, val_loader)
    return checkpoint_callback
