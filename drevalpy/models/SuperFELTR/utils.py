import os
import random
import torch
from torch import nn
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar

from ..MOLIR.utils import generate_triplets_indices, create_dataset_and_loaders


class SuperFELTEncoder(pl.LightningModule):
    def __init__(self, input_size, hpams, omic_type, ranges):
        super(SuperFELTEncoder, self).__init__()
        self.omic_type = omic_type
        if omic_type == 'expression':
            output_size = hpams["out_dim_expr_encoder"]
        elif omic_type == 'mutation':
            output_size = hpams["out_dim_mutation_encoder"]
        elif omic_type == 'copy_number_variation_gistic':
            output_size = hpams["out_dim_cnv_encoder"]
        else:
            raise ValueError(f"omic_type {omic_type} not recognized.")

        # only change vs MOLIEncoder: BatchNorm1d before ReLU
        self.encode = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(hpams["dropout_rate"])
        )
        self.lr = hpams["learning_rate"]
        self.weight_decay = hpams["weight_decay"]
        self.triplet_loss = nn.TripletMarginLoss(margin=hpams["margin"], p=2)
        self.positive_range, self.negative_range = ranges

    def forward(self, x):
        return self.encode(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer

    def get_omic_data(self, data_expr, data_mut, data_cnv):
        if self.omic_type == 'expression':
            data = data_expr
        elif self.omic_type == 'mutation':
            data = data_mut
        elif self.omic_type == 'copy_number_variation_gistic':
            data = data_cnv
        else:
            raise ValueError(f"omic_type {self.omic_type} not recognized.")
        return data

    def compute_loss(self, encoded, response):
        positive_indices, negative_indices = generate_triplets_indices(
            response.cpu().detach().numpy(), self.positive_range, self.negative_range
        )
        triplet_loss = self.triplet_loss(encoded, encoded[positive_indices], encoded[negative_indices])
        return triplet_loss

    def training_step(self, batch, batch_idx):
        data_expr, data_mut, data_cnv, response = batch
        data = self.get_omic_data(data_expr, data_mut, data_cnv)
        encoded = self.encode(data)
        triplet_loss = self.compute_loss(encoded, response)
        self.log("train_loss", triplet_loss, on_step=False, on_epoch=True, prog_bar=True)
        return triplet_loss

    def validation_step(self, batch, batch_idx):
        data_expr, data_mut, data_cnv, response = batch
        data = self.get_omic_data(data_expr, data_mut, data_cnv)
        encoded = self.encode(data)
        triplet_loss = self.compute_loss(encoded, response)
        self.log("val_loss", triplet_loss, on_step=False, on_epoch=True, prog_bar=True)
        return triplet_loss


class SuperFELTRegressor(pl.LightningModule):
    def __init__(self, input_size, hpams, encoders, ranges):
        super(SuperFELTRegressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Dropout(hpams["dropout_rate"])
        )
        self.lr = hpams["learning_rate"]
        self.weight_decay = hpams["weight_decay"]
        self.expr_encoder, self.mut_encoder, self.cnv_encoder = encoders
        self.positive_ranges, self.negative_ranges = ranges
        # put the encoders in eval mode
        self.expr_encoder.eval()
        self.mut_encoder.eval()
        self.cnv_encoder.eval()
        self.regression_loss = nn.MSELoss()

    def forward(self, x):
        return self.regressor(x)

    def predict(self, data_expr, data_mut, data_cnv):
        data_expr = torch.from_numpy(data_expr).float().to(self.device)
        data_mut = torch.from_numpy(data_mut).float().to(self.device)
        data_cnv = torch.from_numpy(data_cnv).float().to(self.device)
        self.eval()
        with torch.no_grad():
            encoded = self.encode_and_concatenate(data_expr, data_mut, data_cnv)
            preds = self.regressor(encoded)
        return preds.squeeze().cpu().detach().numpy()

    def configure_optimizers(self):
        return torch.optim.Adagrad(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def encode_and_concatenate(self, data_expr, data_mut, data_cnv):
        encoded_expr = self.expr_encoder.encode(data_expr)
        encoded_mut = self.mut_encoder.encode(data_mut)
        encoded_cnv = self.cnv_encoder.encode(data_cnv)
        return torch.cat((encoded_expr, encoded_mut, encoded_cnv), dim=1)

    def training_step(self, batch, batch_idx):
        data_expr, data_mut, data_cnv, response = batch
        encoded = self.encode_and_concatenate(data_expr, data_mut, data_cnv)
        pred = self.regressor(encoded)
        loss = self.regression_loss(pred.squeeze(), response)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data_expr, data_mut, data_cnv, response = batch
        encoded = self.encode_and_concatenate(data_expr, data_mut, data_cnv)
        pred = self.regressor(encoded)
        loss = self.regression_loss(pred.squeeze(), response)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


def train_superfeltr_model(model, hpams, output_train, cell_line_input, output_earlystopping, patience=5):
    batch_size = hpams["mini_batch"]
    epochs = hpams["epochs"]
    train_loader, val_loader = create_dataset_and_loaders(
        batch_size=batch_size,
        output_train=output_train,
        cell_line_input=cell_line_input,
        output_earlystopping=output_earlystopping,
    )
    monitor = "train_loss" if (val_loader is None) else "val_loss"
    early_stop_callback = EarlyStopping(
        monitor=monitor, mode="min", patience=patience
    )
    name = "version-" + "".join(
        [random.choice("0123456789abcdef") for i in range(20)]
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
        max_epochs=epochs,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            TQDMProgressBar(),
        ],
        default_root_dir=os.path.join(
            os.getcwd(), "superfeltr_checkpoints/lightning_logs/" + name
        ),
    )
    if val_loader is None:
        trainer.fit(model, train_loader)
    else:
        trainer.fit(model, train_loader, val_loader)
    return checkpoint_callback

