import os
import random
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from torch import nn

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..MOLIR.utils import create_dataset_and_loaders, generate_triplets_indices


class SuperFELTEncoder(pl.LightningModule):
    def __init__(
        self, input_size: int, hpams: dict[str, Union[int, float]], omic_type: str, ranges: tuple[float, float]
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.omic_type = omic_type
        output_size = self.get_output_size(hpams)

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
        return self.encode(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def get_output_size(self, hpams: dict[str, Union[int, float]]) -> int:
        return {
            "expression": hpams["out_dim_expr_encoder"],
            "mutation": hpams["out_dim_mutation_encoder"],
            "copy_number_variation_gistic": hpams["out_dim_cnv_encoder"],
        }[self.omic_type]

    def get_omic_data(self, data_expr: torch.Tensor, data_mut: torch.Tensor, data_cnv: torch.Tensor) -> torch.Tensor:
        if self.omic_type == "expression":
            data = data_expr
        elif self.omic_type == "mutation":
            data = data_mut
        elif self.omic_type == "copy_number_variation_gistic":
            data = data_cnv
        else:
            raise ValueError(f"omic_type {self.omic_type} not recognized.")
        return data

    def compute_loss(self, encoded: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        positive_indices, negative_indices = generate_triplets_indices(
            response.cpu().detach().numpy(), self.positive_range, self.negative_range
        )
        triplet_loss = self.triplet_loss(encoded, encoded[positive_indices], encoded[negative_indices])
        return triplet_loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        data_expr, data_mut, data_cnv, response = batch
        data = self.get_omic_data(data_expr, data_mut, data_cnv)
        encoded = self.encode(data)
        triplet_loss = self.compute_loss(encoded, response)
        self.log("train_loss", triplet_loss, on_step=False, on_epoch=True, prog_bar=True)
        return triplet_loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        data_expr, data_mut, data_cnv, response = batch
        data = self.get_omic_data(data_expr, data_mut, data_cnv)
        encoded = self.encode(data)
        triplet_loss = self.compute_loss(encoded, response)
        self.log("val_loss", triplet_loss, on_step=False, on_epoch=True, prog_bar=True)
        return triplet_loss


class SuperFELTRegressor(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hpams: dict[str, Union[int, float]],
        encoders: tuple[SuperFELTEncoder, SuperFELTEncoder, SuperFELTEncoder],
        ranges: tuple[float, float],
    ) -> None:
        super().__init__()

        self.regressor = nn.Sequential(nn.Linear(input_size, 1), nn.Dropout(hpams["dropout_rate"]))
        self.lr = hpams["learning_rate"]
        self.weight_decay = hpams["weight_decay"]
        self.encoders = encoders
        self.positive_ranges, self.negative_ranges = ranges
        # put the encoders in eval mode
        for encoder in self.encoders:
            encoder.eval()
        self.regression_loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)

    def predict(self, data_expr: np.ndarray, data_mut: np.ndarray, data_cnv: np.ndarray) -> np.ndarray:
        data_expr, data_mut, data_cnv = map(
            lambda data: torch.from_numpy(data).float().to(self.device), [data_expr, data_mut, data_cnv]
        )
        self.eval()
        with torch.no_grad():
            encoded = self.encode_and_concatenate(data_expr, data_mut, data_cnv)
            preds = self.regressor(encoded)
        return preds.squeeze().cpu().detach().numpy()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adagrad(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def encode_and_concatenate(
        self, data_expr: torch.Tensor, data_mut: torch.Tensor, data_cnv: torch.Tensor
    ) -> torch.Tensor:
        encoded_expr = self.encoders[0].encode(data_expr)
        encoded_mut = self.encoders[1].encode(data_mut)
        encoded_cnv = self.encoders[2].encode(data_cnv)
        return torch.cat((encoded_expr, encoded_mut, encoded_cnv), dim=1)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        data_expr, data_mut, data_cnv, response = batch
        encoded = self.encode_and_concatenate(data_expr, data_mut, data_cnv)
        pred = self.regressor(encoded)
        loss = self.regression_loss(pred.squeeze(), response)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        data_expr, data_mut, data_cnv, response = batch
        encoded = self.encode_and_concatenate(data_expr, data_mut, data_cnv)
        pred = self.regressor(encoded)
        loss = self.regression_loss(pred.squeeze(), response)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


def train_superfeltr_model(
    model: Union[SuperFELTEncoder, SuperFELTRegressor],
    hpams: dict[str, Union[int, float]],
    output_train: DrugResponseDataset,
    cell_line_input: FeatureDataset,
    output_earlystopping: DrugResponseDataset,
    patience: int = 5,
) -> pl.callbacks.ModelCheckpoint:
    train_loader, val_loader = create_dataset_and_loaders(
        batch_size=hpams["mini_batch"],
        output_train=output_train,
        cell_line_input=cell_line_input,
        output_earlystopping=output_earlystopping,
    )
    monitor = "train_loss" if (val_loader is None) else "val_loss"
    early_stop_callback = EarlyStopping(monitor=monitor, mode="min", patience=patience)
    name = "version-" + "".join(
        [random.choice("0123456789abcdef") for _ in range(20)]
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
