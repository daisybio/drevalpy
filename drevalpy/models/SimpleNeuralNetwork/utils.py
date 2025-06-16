"""Utility functions for the simple neural network models."""

import os
import secrets

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
        drug_input: FeatureDataset,
        cell_line_views: list[str],
        drug_views: list[str],
    ):
        """
        Initializes the regression dataset.

        :param output: response values
        :param cell_line_input: input omics data
        :param drug_input: input fingerprint data
        :param cell_line_views: either gene expression for the SimpleNeuralNetwork or all omics data for the
            MultiOMICSNeuralNetwork
        :param drug_views: fingerprints
        :raises AssertionError: if the views are not found in the input data
        """
        self.cell_line_views = cell_line_views
        self.drug_views = drug_views
        self.output = output
        self.cell_line_input = cell_line_input
        self.drug_input = drug_input
        for cl_view in self.cell_line_views:
            if cl_view not in cell_line_input.view_names:
                raise AssertionError(f"Cell line view {cl_view} not found in cell line input")
        for d_view in self.drug_views:
            if d_view not in drug_input.view_names:
                raise AssertionError(f"Drug view {d_view} not found in drug input")

    def __getitem__(self, idx):
        """
        Overwrites the getitem method from the Dataset class.

        Retrieves the cell line and drug features and the response for the given index.
        :param idx: index of the sample of interest
        :returns: the cell line feature(s) and the response
        :raises TypeError: if the features are not numpy arrays
        """
        cell_line_id = self.output.cell_line_ids[idx]
        drug_id = self.output.drug_ids[idx]
        response = self.output.response[idx]
        cell_line_features = None
        drug_features = None
        for cl_view in self.cell_line_views:
            feature_mat = self.cell_line_input.features[cell_line_id][cl_view]

            if cell_line_features is None:
                cell_line_features = feature_mat
            else:
                cell_line_features = np.concatenate((cell_line_features, feature_mat))
        for d_view in self.drug_views:
            if drug_features is None:
                drug_features = self.drug_input.features[drug_id][d_view]
            else:
                drug_features = np.concatenate((drug_features, self.drug_input.features[drug_id][d_view]))
        if not isinstance(cell_line_features, np.ndarray):
            raise TypeError(f"Cell line features for {cell_line_id} are not numpy array")
        if not isinstance(drug_features, np.ndarray):
            raise TypeError(f"Drug features for {drug_id} are not numpy array")
        data = np.concatenate((cell_line_features, drug_features))
        # cast to float32
        data = data.astype(np.float32)
        response = np.float32(response)
        return data, response

    def __len__(self):
        """
        Overwrites the len method from the Dataset class.

        :returns: the length of the output
        """
        return len(self.output.response)


class FeedForwardNetwork(pl.LightningModule):
    """Feed forward neural network for regression tasks with basic architecture."""

    def __init__(self, hyperparameters: dict[str, int | float | list[int]], input_dim: int) -> None:
        """
        Initializes the feed forward network.

        The model uses a simple architecture with fully connected layers, batch normalization, and dropout. An MSE
        loss is used.

        :param hyperparameters: hyperparameters
        :param input_dim: input dimension, for SimpleNeuralNetwork it is the sum of the gene expression and
            fingerprint, for MultiOMICSNeuralNetwork it is the sum of all omics data and fingerprints
        :raises TypeError: if the hyperparameters are not of the correct type
        """
        super().__init__()
        self.save_hyperparameters()

        if not isinstance(hyperparameters["units_per_layer"], list):
            raise TypeError("units_per_layer must be a list of integers")
        if not isinstance(hyperparameters["dropout_prob"], float):
            raise TypeError("dropout_prob must be a float")

        n_units_per_layer: list[int] = hyperparameters["units_per_layer"]
        dropout_prob: float = hyperparameters["dropout_prob"]
        self.n_units_per_layer = n_units_per_layer
        self.dropout_prob = dropout_prob
        self.loss = nn.MSELoss()
        # self.checkpoint_callback is initialized in the fit method
        self.checkpoint_callback: pl.callbacks.ModelCheckpoint | None = None
        self.fully_connected_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        self.dropout_layer = None

        self.fully_connected_layers.append(nn.Linear(input_dim, self.n_units_per_layer[0]))
        self.batch_norm_layers.append(nn.BatchNorm1d(self.n_units_per_layer[0]))

        for i in range(1, len(self.n_units_per_layer)):
            self.fully_connected_layers.append(nn.Linear(self.n_units_per_layer[i - 1], self.n_units_per_layer[i]))
            self.batch_norm_layers.append(nn.BatchNorm1d(self.n_units_per_layer[i]))

        self.fully_connected_layers.append(nn.Linear(self.n_units_per_layer[-1], 1))
        if self.dropout_prob is not None:
            self.dropout_layer = nn.Dropout(p=self.dropout_prob)

    def fit(
        self,
        output_train: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None,
        cell_line_views: list[str],
        drug_views: list[str],
        output_earlystopping: DrugResponseDataset | None = None,
        trainer_params: dict | None = None,
        batch_size=32,
        patience=5,
        num_workers: int = 2,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Fits the model.

        First, the data is loaded using a DataLoader. Then, the model is trained using the Lightning Trainer.
        :param output_train: Response values for training
        :param cell_line_input: Cell line features
        :param drug_input: Drug features
        :param cell_line_views: Cell line info needed for this model
        :param drug_views: Drug info needed for this model
        :param output_earlystopping: Response values for early stopping
        :param trainer_params: custom parameters for the trainer
        :param batch_size: batch size for the DataLoader, default is 32
        :param patience: patience for early stopping, default is 5
        :param num_workers: number of workers for the DataLoader, default is 2
        :param model_checkpoint_dir: directory to save the model checkpoints
        :raises ValueError: if drug_input is missing
        """
        if trainer_params is None:
            trainer_params = {
                "max_epochs": 100,
                "progress_bar_refresh_rate": 500,
            }
        if drug_input is None:
            raise ValueError(
                "Drug input (fingerprints) are required for SimpleNeuralNetwork and " "MultiOMICsNeuralNetwork."
            )

        train_dataset = RegressionDataset(
            output=output_train,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            cell_line_views=cell_line_views,
            drug_views=drug_views,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
            drop_last=True,  # to avoid batch norm errors, if last batch is smaller than batch_size, it is not processed
        )

        val_loader = None
        if output_earlystopping is not None:
            val_dataset = RegressionDataset(
                output=output_earlystopping,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
                cell_line_views=cell_line_views,
                drug_views=drug_views,
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

        early_stop_callback = EarlyStopping(monitor=monitor, mode="min", patience=patience)

        unique_subfolder = os.path.join(model_checkpoint_dir, "run_" + secrets.token_hex(8))
        os.makedirs(unique_subfolder, exist_ok=True)

        # prevent conflicts
        name = "version-" + "".join([secrets.choice("0123456789abcdef") for _ in range(10)])
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=unique_subfolder,
            monitor=monitor,
            mode="min",
            save_top_k=1,
            filename=name,
        )

        progress_bar = TQDMProgressBar(refresh_rate=trainer_params["progress_bar_refresh_rate"])
        trainer_params_copy = trainer_params.copy()
        del trainer_params_copy["progress_bar_refresh_rate"]

        # Initialize the Lightning trainer
        trainer = pl.Trainer(
            callbacks=[
                early_stop_callback,
                self.checkpoint_callback,
                progress_bar,
            ],
            default_root_dir=model_checkpoint_dir,
            devices=1,
            **trainer_params_copy,
        )
        if val_loader is None:
            trainer.fit(self, train_loader)
        else:
            trainer.fit(self, train_loader, val_loader)

        # load best model
        if self.checkpoint_callback.best_model_path is not None:
            checkpoint = torch.load(self.checkpoint_callback.best_model_path, weights_only=True)  # noqa: S614
            self.load_state_dict(checkpoint["state_dict"])
        else:
            print("checkpoint_callback: No best model found, using the last model.")

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: input data
        :returns: predicted response
        """
        for i in range(len(self.fully_connected_layers) - 2):
            x = self.fully_connected_layers[i](x)
            x = self.batch_norm_layers[i](x)
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)
            x = torch.relu(x)

        x = torch.relu(self.fully_connected_layers[-2](x))
        x = self.fully_connected_layers[-1](x)

        return x.squeeze()

    def _forward_loss_and_log(self, x, y, log_as: str):
        """
        Forward pass, calculates the loss, and logs the loss.

        :param x: input data
        :param y: response
        :param log_as: either train_loss or val_loss
        :returns: loss
        """
        y_pred = self.forward(x)
        result = self.loss(y_pred, y)
        self.log(log_as, result, on_step=True, on_epoch=True, prog_bar=True)
        return result

    def training_step(self, batch):
        """
        Overwrites the training step from the LightningModule.

        Does a forward pass, calculates the loss and logs the loss.
        :param batch: batch of data
        :returns: loss
        """
        x, y = batch
        return self._forward_loss_and_log(x, y, "train_loss")

    def validation_step(self, batch):
        """
        Overwrites the validation step from the LightningModule.

        Does a forward pass, calculates the loss and logs the loss.
        :param batch: batch of data
        :returns: loss
        """
        x, y = batch
        return self._forward_loss_and_log(x, y, "val_loss")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the response for the given input.

        :param x: input data
        :returns: predicted response
        """
        is_training = self.training
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(torch.from_numpy(x).float().to(self.device))
        self.train(is_training)
        return y_pred.cpu().detach().numpy()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Overwrites the configure_optimizers from the LightningModule.

        :returns: Adam optimizer
        """
        return torch.optim.Adam(self.parameters())
