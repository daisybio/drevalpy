"""Lightning network used by the dense neural-network predictor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn

from drevalpy.components.lightning_metrics_mixin import RegressionMetricsMixin


class FeedForwardNetwork(RegressionMetricsMixin, pl.LightningModule):
    """Feed-forward regression network with batch normalization and dropout."""

    def __init__(self, hyperparameters: dict[str, Any], input_dim: int) -> None:
        """Initialize instance state.

        :param hyperparameters: hyperparameters.
        :param input_dim: input dim.
        :raises TypeError: Raised on invalid input.
        """
        super().__init__()
        self.save_hyperparameters()

        units_per_layer = hyperparameters["units_per_layer"]
        if not isinstance(units_per_layer, list) or not all(isinstance(unit, int) for unit in units_per_layer):
            msg = "units_per_layer must be a list of integers"
            raise TypeError(msg)
        dropout_prob = hyperparameters["dropout_prob"]
        if not isinstance(dropout_prob, float):
            msg = "dropout_prob must be a float"
            raise TypeError(msg)

        self.n_units_per_layer: list[int] = units_per_layer
        self.dropout_prob: float = dropout_prob
        self.loss = nn.MSELoss()
        self.fully_connected_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.dropout_layer: nn.Dropout | None = None

        self.fully_connected_layers.append(nn.Linear(input_dim, self.n_units_per_layer[0]))
        self.batch_norm_layers.append(nn.BatchNorm1d(self.n_units_per_layer[0]))
        for index in range(1, len(self.n_units_per_layer)):
            self.fully_connected_layers.append(
                nn.Linear(self.n_units_per_layer[index - 1], self.n_units_per_layer[index])
            )
            self.batch_norm_layers.append(nn.BatchNorm1d(self.n_units_per_layer[index]))
        self.fully_connected_layers.append(nn.Linear(self.n_units_per_layer[-1], 1))
        self.dropout_layer = nn.Dropout(p=self.dropout_prob)
        self._init_metrics_storage()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict responses from a batch of concatenated feature rows.

        :param features: features.
        :returns: Result.
        """
        hidden = features
        for index in range(len(self.fully_connected_layers) - 2):
            hidden = self.fully_connected_layers[index](hidden)
            hidden = self.batch_norm_layers[index](hidden)
            if self.dropout_layer is not None:
                hidden = self.dropout_layer(hidden)
            hidden = torch.relu(hidden)
        hidden = torch.relu(self.fully_connected_layers[-2](hidden))
        return self.fully_connected_layers[-1](hidden).squeeze(dim=-1)

    def _loss_and_log(self, features: torch.Tensor, response: torch.Tensor, name: str) -> torch.Tensor:
        predictions = self(features)
        loss = self.loss(predictions, response)
        self.log(name, loss, on_step=True, on_epoch=True, prog_bar=True)
        self._store_predictions(predictions, response, is_training=name == "train_loss")
        return loss

    @staticmethod
    def _unpack_batch(batch: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate feature tensors and separate the response.

        The batch is ``(*feature_tensors, response)`` where the number of feature
        tensors depends on how many entity blocks are present (e.g. cell-line
        only vs cell-line + drug).

        :param batch: Sequence of tensors from the DataLoader.
        :returns: ``(concatenated_features, response)``.
        """
        features = torch.cat(list(batch[:-1]), dim=1)
        return features, batch[-1]

    def training_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute and log training loss for one batch.

        :param batch: batch.
        :param batch_idx: batch idx.
        :returns: Result.
        """
        _ = batch_idx
        features, response = self._unpack_batch(batch)
        return self._loss_and_log(features, response, name="train_loss")

    def validation_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute and log validation loss for one batch.

        :param batch: batch.
        :param batch_idx: batch idx.
        :returns: Result.
        """
        _ = batch_idx
        features, response = self._unpack_batch(batch)
        return self._loss_and_log(features, response, name="val_loss")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Build the Adam optimizer used by the original predictor.

        :returns: Result.
        """
        return torch.optim.Adam(self.parameters())
