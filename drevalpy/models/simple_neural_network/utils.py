"""
Utility functions for the simple neural network models.
"""

import os
import random
from typing import Optional, List
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset


class RegressionDataset(Dataset):
    """
    Dataset for regression tasks for the data loader.
    """

    def __init__(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset = None,
        drug_input: FeatureDataset = None,
        cell_line_views: List[str] = None,
        drug_views: List[str] = None,
        met_transform=None,
    ):
        self.cell_line_views = cell_line_views
        self.drug_views = drug_views
        self.output = output
        self.cell_line_input = cell_line_input
        self.drug_input = drug_input
        for cl_view in self.cell_line_views:
            assert (
                cl_view in cell_line_input.view_names
            ), f"Cell line view {cl_view} not found in cell line input"
        for d_view in self.drug_views:
            assert (
                d_view in drug_input.view_names
            ), f"Drug view {d_view} not found in drug input"
        self.met_transform = met_transform

    def __getitem__(self, idx):
        cell_line_id = self.output.cell_line_ids[idx]
        drug_id = self.output.drug_ids[idx]
        response = self.output.response[idx]
        cell_line_features = None
        drug_features = None
        for cl_view in self.cell_line_views:
            feature_mat = self.cell_line_input.features[cell_line_id][cl_view]
            if cl_view == "methylation" and self.met_transform is not None:
                # reshape because it contains a single sample
                feature_mat = feature_mat.reshape(1, -1)
                feature_mat = self.met_transform.transform(feature_mat)
                # reshape back to original shape
                feature_mat = feature_mat.reshape(-1)
            if cell_line_features is None:
                cell_line_features = feature_mat
            else:
                cell_line_features = np.concatenate((cell_line_features, feature_mat))
        for d_view in self.drug_views:
            if drug_features is None:
                drug_features = self.drug_input.features[drug_id][d_view]
            else:
                drug_features = np.concatenate(
                    (drug_features, self.drug_input.features[drug_id][d_view])
                )
        assert isinstance(
            cell_line_features, np.ndarray
        ), f"Cell line features for {cell_line_id} are not numpy array"
        assert isinstance(
            drug_features, np.ndarray
        ), f"Drug features for {drug_id} are not numpy array"
        data = np.concatenate((cell_line_features, drug_features))
        # cast to float32
        data = data.astype(np.float32)
        response = np.float32(response)
        return data, response

    def __len__(self):
        return len(self.output.response)


class FeedForwardNetwork(pl.LightningModule):
    """
    Feed forward neural network for regression tasks with basic architecture.
    """

    def __init__(self, n_units_per_layer=None, dropout_prob=None) -> None:
        super().__init__()
        if n_units_per_layer is None:
            n_units_per_layer = [256, 64]
        self.n_units_per_layer = n_units_per_layer
        self.dropout_prob = dropout_prob
        self.model_initialized = False
        self.loss = nn.MSELoss()
        self.checkpoint_callback = None
        self.fully_connected_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.dropout_layer = None

    def fit(
        self,
        output_train: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset = None,
        cell_line_views: List[str] = None,
        drug_views: List[str] = None,
        output_earlystopping: Optional[DrugResponseDataset] = None,
        trainer_params: Optional[dict] = None,
        batch_size=32,
        patience=5,
        checkpoint_path: Optional[str] = None,
        num_workers: int = 2,
        met_transform=None,
    ) -> None:
        """
        Fits the model.
        :param output_train: Response values for training
        :param cell_line_input: Cell line features
        :param drug_input: Drug features
        :param cell_line_views: Cell line info needed for this model
        :param drug_views: Drug info needed for this model
        :param output_earlystopping: Response values for early stopping
        :param trainer_params: custom parameters for the trainer
        :param batch_size:
        :param patience:
        :param checkpoint_path:
        :param num_workers:
        :param met_transform:
        :return:
        """
        if trainer_params is None:
            trainer_params = {"progress_bar_refresh_rate": 300, "max_epochs": 70}

        train_dataset = RegressionDataset(
            output=output_train,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            cell_line_views=cell_line_views,
            drug_views=drug_views,
            met_transform=met_transform,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
        )

        val_loader = None
        if output_earlystopping is not None:
            val_dataset = RegressionDataset(
                output=output_earlystopping,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
                cell_line_views=cell_line_views,
                drug_views=drug_views,
                met_transform=met_transform,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=True,
            )

        # Train the model
        monitor = "train_loss" if (val_loader is None) else "val_loss"

        early_stop_callback = EarlyStopping(
            monitor=monitor, mode="min", patience=patience
        )
        name = "version-" + "".join(
            [random.choice("0123456789abcdef") for i in range(20)]
        )  # preventing conflicts of filenames
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_path,
            monitor=monitor,
            mode="min",
            save_top_k=1,
            filename=name,
        )

        progress_bar = TQDMProgressBar(
            refresh_rate=trainer_params["progress_bar_refresh_rate"]
        )
        trainer_params_copy = trainer_params.copy()
        del trainer_params_copy["progress_bar_refresh_rate"]

        # Force initialize model with dummy data
        self.force_initialize(train_loader)

        # Initialize the Lightning trainer
        trainer = pl.Trainer(
            callbacks=[early_stop_callback, self.checkpoint_callback, progress_bar],
            default_root_dir=os.path.join(
                os.getcwd(), "model_checkpoints/lightning_logs/" + name
            ),
            **trainer_params_copy,
        )
        if val_loader is None:

            trainer.fit(self, train_loader)
        else:
            trainer.fit(self, train_loader, val_loader)
        # TODO use best model from history self.load_from_checkpoint(
        #  self.checkpoint_callback.best_model_path)

    def forward(self, x):
        """
        Forward pass of the model.
        :param x:
        :return:
        """
        if not self.model_initialized:
            self.initialize_model(x)

        for i in range(len(self.fully_connected_layers) - 2):
            x = self.fully_connected_layers[i](x)
            x = self.batch_norm_layers[i](x)
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)
            x = torch.relu(x)

        x = torch.relu(self.fully_connected_layers[-2](x))
        x = self.fully_connected_layers[-1](x)

        return x.squeeze()

    def initialize_model(self, x):
        """
        Initializes the model.
        :param x:
        :return:
        """
        n_features = x.size(1)
        self.fully_connected_layers.append(
            nn.Linear(n_features, self.n_units_per_layer[0])
        )
        self.batch_norm_layers.append(nn.BatchNorm1d(self.n_units_per_layer[0]))

        for i in range(1, len(self.n_units_per_layer)):
            self.fully_connected_layers.append(
                nn.Linear(self.n_units_per_layer[i - 1], self.n_units_per_layer[i])
            )
            self.batch_norm_layers.append(nn.BatchNorm1d(self.n_units_per_layer[i]))

        self.fully_connected_layers.append(nn.Linear(self.n_units_per_layer[-1], 1))
        if self.dropout_prob is not None:
            self.dropout_layer = nn.Dropout(p=self.dropout_prob)
        self.model_initialized = True

    def force_initialize(self, dataloader):
        """Force initialize the model by running a dummy forward pass."""
        for batch in dataloader:
            x, _ = batch
            self.forward(x)
            break

    def _forward_loss_and_log(self, x, y, log_as: str):
        y_pred = self.forward(x)
        result = self.loss(y_pred, y)
        self.log(log_as, result, on_step=True, on_epoch=True, prog_bar=True)
        return result

    def training_step(self, batch):
        x, y = batch
        return self._forward_loss_and_log(x, y, "train_loss")

    def validation_step(self, batch):
        x, y = batch
        return self._forward_loss_and_log(x, y, "val_loss")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the response for the given input.
        :param x:
        :return:
        """
        is_training = self.training
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(torch.from_numpy(x).float())
        self.train(is_training)
        return y_pred.cpu().detach().numpy()

    def configure_optimizers(self):

        return torch.optim.Adam(self.parameters())
