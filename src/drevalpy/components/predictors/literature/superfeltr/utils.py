"""Utility functions for the SuperFELTR model."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytorch_lightning as pl
import torch
from torch import nn

from drevalpy.components.predictors._lightning_training import LossSteps
from drevalpy.components.predictors.literature.molir.utils import OmicsRegressionModule, generate_triplets_indices

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The encoder omic types in the order of a batch's three omic views, each with the hyperparameter holding
#: its encoder's output width.
ENCODER_DIM_KEYS = {
    "expression": "out_dim_expr_encoder",
    "mutation": "out_dim_mutation_encoder",
    "copy_number_variation_gistic": "out_dim_cnv_encoder",
}


class SuperFELTEncoder(LossSteps, pl.LightningModule):
    """SuperFELT encoder definition for a single omic type, i.e., gene expression, mutation, or copy number variation.

    Very similar to MOLIEncoder, but with BatchNorm1d before ReLU.
    """

    def __init__(
        self, input_size: int, hpams: dict[str, int | float | dict], omic_type: str, ranges: tuple[float, float]
    ) -> None:
        """Initialize the SuperFELTEncoder.

        Save_hyperparameters is turned on to facilitate loading the model from a checkpoint.

        :param input_size: determined by the variance threshold feature selection
        :param hpams: hyperparameters for the model
        :param omic_type: one of the keys of ``ENCODER_DIM_KEYS`` (a ``KeyError`` for any other)
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

        if not all(isinstance(hpams[key], int) for key in ENCODER_DIM_KEYS.values()):
            raise ValueError("out_dim_expr_encoder, out_dim_mutation_encoder, and out_dim_cnv_encoder must be ints!")

        self.omic_type = omic_type
        output_size = hpams[ENCODER_DIM_KEYS[omic_type]]

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

        :param x: Omic features of shape [batch, input_size].

        :returns: Encoded features of shape [batch, output_size].
        """
        return self.encode(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Override the configure_optimizers method to use the Adam optimizer.

        :returns: Adam optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _compute_loss(self, encoded: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        """Compute the triplet loss.

        :param encoded: encoded data
        :param response: response data

        :returns: triplet loss
        """
        positive_indices, negative_indices = generate_triplets_indices(
            response.cpu().detach().numpy(), self.positive_range, self.negative_range
        )
        triplet_loss = self.triplet_loss(encoded, encoded[positive_indices], encoded[negative_indices])
        return triplet_loss

    def _loss_and_log(self, batch: Sequence[torch.Tensor], key: str) -> torch.Tensor:
        """Compute and log the triplet loss of one batch on this encoder's omic view.

        :param batch: batch containing the omic data and response
        :param key: name the loss is logged under

        :returns: triplet loss
        """
        *views, response = batch
        response = response.reshape(-1)
        encoded = self.encode(views[list(ENCODER_DIM_KEYS).index(self.omic_type)])
        triplet_loss = self._compute_loss(encoded, response)
        self.log(key, triplet_loss, on_step=False, on_epoch=True, prog_bar=True)
        return triplet_loss


class SuperFELTRegressor(LossSteps, OmicsRegressionModule):
    """SuperFELT regressor definition.

    Very similar to SuperFELT classifier, but with a regression loss and without the last sigmoid layer.
    """

    def __init__(
        self,
        input_size: int,
        hpams: dict[str, int | float | dict],
        encoders: tuple[SuperFELTEncoder, SuperFELTEncoder, SuperFELTEncoder],
    ) -> None:
        """Initialize the SuperFELTRegressor.

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
        # Registered as a ModuleList so Lightning's device placement reaches the
        # encoders too; a plain tuple left them behind on CPU while the regressor
        # moved to the accelerator.
        self.encoders = nn.ModuleList(encoders)
        # The encoders were fitted beforehand, so freeze them and keep them in
        # eval mode: their BatchNorm running stats and Dropout must stay fixed.
        for encoder in self.encoders:
            encoder.eval()
            encoder.requires_grad_(False)
        self.regression_loss = nn.MSELoss()

    def train(self, mode: bool = True) -> SuperFELTRegressor:
        """Set training mode on the regressor while keeping the encoders in eval mode.

        Lightning calls ``train()`` when the fit loop starts, which would otherwise
        reactivate the frozen encoders' BatchNorm updates and Dropout.

        :param mode: Whether to put the regressor into training mode.
        :returns: This regressor, so calls can be chained.
        """
        super().train(mode)
        for encoder in self.encoders:
            encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SuperFELTRegressor.

        :param x: Concatenated encoder outputs of shape [batch, input_size].

        :returns: Predicted response of shape [batch, 1].
        """
        return self.regressor(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Override the configure_optimizers method to use the Adagrad optimizer.

        Only the regressor head is optimized; the encoders are frozen.

        :returns: Adagrad optimizer
        """
        return torch.optim.Adagrad(self.regressor.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _encoder(self, index: int) -> SuperFELTEncoder:
        """Return the registered encoder at ``index`` with its concrete type.

        ``nn.ModuleList.__getitem__`` is typed as ``Tensor | Module``, so the cast
        keeps the ``encode`` calls below statically checkable.

        :param index: Position of the encoder in the registered list.
        :returns: The encoder at that position.
        """
        return cast("SuperFELTEncoder", self.encoders[index])

    def _encode_and_concatenate(
        self, gene_expression: torch.Tensor, mutations: torch.Tensor, copy_number: torch.Tensor
    ) -> torch.Tensor:
        """Encode the omic data and concatenates the encoded tensors.

        :param gene_expression: Gene expression features of the batch.
        :param mutations: Mutation features of the batch.
        :param copy_number: Copy number variation features of the batch.

        :returns: Encoder outputs concatenated along the feature axis, in the order expression, mutation, copy number.
        """
        encoded_expr = self._encoder(0).encode(gene_expression)
        encoded_mut = self._encoder(1).encode(mutations)
        encoded_cnv = self._encoder(2).encode(copy_number)
        return torch.cat((encoded_expr, encoded_mut, encoded_cnv), dim=1)

    def _loss_and_log(self, batch: Sequence[torch.Tensor], key: str) -> torch.Tensor:
        """Compute and log the regression loss of one batch.

        :param batch: batch containing the omic data and response
        :param key: name the loss is logged under

        :returns: regression loss
        """
        data_expr, data_mut, data_cnv, response = batch
        response = response.reshape(-1)
        encoded = self._encode_and_concatenate(data_expr, data_mut, data_cnv)
        pred = self.regressor(encoded)
        loss = self.regression_loss(pred.squeeze(), response)
        self.log(key, loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
