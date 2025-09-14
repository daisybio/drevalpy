"""Implements the PaccMann model."""

from typing import Any

import numpy as np
import torch

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..drp_model import DRPModel
from ..utils import load_and_select_gene_features, load_drug_smiles_from_csv

try:
    from .paccmann_predictor.paccmann_predictor.models.paccmann import MCA
    from .paccmann_predictor.paccmann_predictor.utils.hyperparams import (
        OPTIMIZER_FACTORY,
    )
except ImportError:
    raise ImportError("PaccMann requires pytoda. Please install it with `pip install .[paccmann]`")


class PaccMann(DRPModel):
    """PaccMann model."""

    def get_model_name(self) -> str:
        """Return the name of the model.

        :return: Name of the model.
        """
        return "PaccMann"

    @property
    def cell_line_views(self) -> list[str]:
        """Return the sources the model needs as input for describing the cell line.

        :return: List of sources.
        """
        return ["gene_expression"]

    @property
    def drug_views(self) -> list[str]:
        """Return the sources the model needs as input for describing the drug.

        :return: List of sources.
        """
        return ["smiles"]

    def load_cell_line_features(self, data_path: str, dataset_name: str, cell_line_view_name: str) -> FeatureDataset:
        """Load the cell line features.

        :param data_path: Path to the data.
        :param dataset_name: Name of the dataset.
        :param cell_line_view_name: Name of the cell line view.
        :return: Feature dataset.
        """
        return load_and_select_gene_features(data_path, dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str, drug_view_name: str) -> FeatureDataset:
        """Load the drug features.

        :param data_path: Path to the data.
        :param dataset_name: Name of the dataset.
        :param drug_view_name: Name of the drug view.
        :return: Feature dataset.
        """
        return load_drug_smiles_from_csv(data_path, dataset_name)

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """Build the model.

        :param hyperparameters: Hyperparameters for the model.
        """
        self.model = MCA(hyperparameters)
        self.optimizer = OPTIMIZER_FACTORY[hyperparameters.get("optimizer", "adam")](
            self.model.parameters(), lr=hyperparameters.get("lr", 0.001)
        )
        self.epochs = hyperparameters.get("epochs", 10)
        self.batch_size = hyperparameters.get("batch_size", 32)

    def train(self, train_dataset: DrugResponseDataset, val_dataset: DrugResponseDataset = None, **kwargs) -> None:
        """Train the model.

        :param train_dataset: Training dataset.
        :param val_dataset: Validation dataset.
        :param kwargs: Additional arguments.
        """
        self.model.train()

        x_drug_train, x_cell_train, y_train = train_dataset.get_drug_cell_line_response_data(
            drug_view="smiles", cell_line_view="gene_expression"
        )

        for _epoch in range(self.epochs):
            for i in range(0, len(x_drug_train), self.batch_size):

                drug_batch = torch.tensor(x_drug_train[i : i + self.batch_size]).to(self.model.device)
                cell_batch = torch.tensor(x_cell_train[i : i + self.batch_size]).to(self.model.device)
                y_batch = torch.tensor(y_train[i : i + self.batch_size]).to(self.model.device)

                self.optimizer.zero_grad()
                y_hat, _, _, _, _ = self.model(drug_batch, cell_batch)
                loss = self.model.loss_fn(y_hat, y_batch)
                loss.backward()
                self.optimizer.step()

    def predict(self, test_dataset: DrugResponseDataset, **kwargs) -> np.ndarray:
        """Predict the response for a given dataset.

        :param test_dataset: Dataset to predict on.
        :param kwargs: Additional arguments.
        :return: Predicted response.
        """
        self.model.eval()

        x_drug_test, x_cell_test, _ = test_dataset.get_drug_cell_line_response_data(
            drug_view="smiles", cell_line_view="gene_expression"
        )

        predictions = []
        with torch.no_grad():
            for i in range(0, len(x_drug_test), self.batch_size):
                drug_batch = torch.tensor(x_drug_test[i : i + self.batch_size]).to(self.model.device)
                cell_batch = torch.tensor(x_cell_test[i : i + self.batch_size]).to(self.model.device)

                y_hat, _, _, _, _ = self.model(drug_batch, cell_batch)
                predictions.append(y_hat.cpu().numpy())

        return np.concatenate(predictions)

    def evaluate(self, test_dataset: DrugResponseDataset, **kwargs) -> dict[str, float]:
        """Evaluate the model.

        :param test_dataset: Dataset to evaluate on.
        :param kwargs: Additional arguments.
        :return: Evaluation metrics.
        """
        _, _, y_true = test_dataset.get_drug_cell_line_response_data(
            drug_view="smiles", cell_line_view="gene_expression"
        )

        y_pred = self.predict(test_dataset, **kwargs)

        return {"mse": float(np.mean((y_true - y_pred) ** 2))}
