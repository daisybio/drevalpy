"""Precily model for drug response prediction.

Contains Precily, a pathway-based deep learning model for drug response
prediction. A deep neural network that predicts LN(IC50) by combining
GSVA pathway-activity scores with SMILESVec drug embeddings.

Original authors: Chawla et al. (2022, 10.1038/s41467-022-33291-z)
Reference code: https://github.com/SmritiChawla/Precily
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from .model_utils import PrecilyNetwork


class _PrecilyDataset(Dataset):
    """PyTorch Dataset yielding (pathway_features, drug_features, response)."""

    def __init__(
        self,
        response: np.ndarray,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_features: FeatureDataset,
        drug_features: FeatureDataset,
    ):
        """Initialize the dataset.

        :param response: drug response values
        :param cell_line_ids: cell line identifiers
        :param drug_ids: drug identifiers
        :param cell_line_features: FeatureDataset with the "pathways" view
        :param drug_features: FeatureDataset with the "smilesvec" view
        """
        self.response = response
        self.cell_line_ids = cell_line_ids
        self.drug_ids = drug_ids
        self.cell_line_features = cell_line_features
        self.drug_features = drug_features

    def __len__(self) -> int:
        """Return the number of samples.

        :returns: Number of samples in the dataset.
        """
        return len(self.response)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample by index.

        :param idx: sample index

        :returns: (pathway_features, drug_features, response) tensors
        """
        cell_line_id = self.cell_line_ids[idx]
        drug_id = self.drug_ids[idx]

        pathway = torch.tensor(self.cell_line_features.features[cell_line_id]["pathways"], dtype=torch.float32)
        drug = torch.tensor(self.drug_features.features[drug_id]["smilesvec"], dtype=torch.float32)
        response = torch.tensor(self.response[idx], dtype=torch.float32)

        return pathway, drug, response


class PrecilyModel(LiteratureTrainingMixin):
    """Precily model for drug response prediction."""

    cell_line_views = ["pathways"]
    drug_views = ["smilesvec"]
    early_stopping = False

    def __init__(self) -> None:
        """Initialize the Precily model."""
        super().__init__()
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: PrecilyNetwork | None = None
        self.hyperparameters: dict[str, Any] = {}

    @classmethod
    def get_model_name(cls) -> str:
        """Get the model name.

        :returns: Precily
        """
        return "Precily"

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, Any]:
        """Return default Precily hyperparameters.

        :returns: Default hyperparameter mapping.
        """
        return {
            "learning_rate": 1.0e-3,
            "dropout": 0.1,
            "epochs": 50,
            "batch_size": 128,
            "seed": 42,
        }

    def configure(self, hyperparameters: dict[str, Any]) -> None:
        """Store hyperparameters.

        The network is built in train() once the input dimension
        (n_pathways + n_drug_features) is known.

        :param hyperparameters: dropout, learning_rate, epochs, batch_size, seed
        """
        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = hyperparameters

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str | Path = "checkpoints",
    ) -> None:
        """Train the Precily model.

        :param output: training response data
        :param cell_line_input: cell line pathway features
        :param drug_input: drug SMILESVec features
        :param output_earlystopping: unused
        :param model_checkpoint_dir: unused

        :raises ValueError: if drug_input is None
        """
        if drug_input is None:
            raise ValueError("Precily model requires drug features.")

        # Resolve input dimension from the feature matrices.
        n_pathways = len(next(iter(cell_line_input.features.values()))["pathways"])
        drug_dim = len(next(iter(drug_input.features.values()))["smilesvec"])
        input_dim = n_pathways + drug_dim

        self.model = PrecilyNetwork(
            input_dim=input_dim,
            dropout=self.hyperparameters.get("dropout", 0.1),
        ).to(self.DEVICE)

        loss_func = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters["learning_rate"])

        train_dataset = _PrecilyDataset(
            response=output.response,
            cell_line_ids=output.cell_line_ids,
            drug_ids=output.drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.hyperparameters["batch_size"],
            shuffle=True,
        )

        print("Training Precily model")
        for epoch in range(self.hyperparameters["epochs"]):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            for pathway_inputs, drug_inputs, targets in train_loader:
                pathway_inputs = pathway_inputs.to(self.DEVICE)
                drug_inputs = drug_inputs.to(self.DEVICE)
                targets = targets.to(self.DEVICE)

                x = torch.cat([pathway_inputs, drug_inputs], dim=1)
                outputs = self.model(x)
                loss = loss_func(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().item()
                batch_count += 1

            epoch_loss /= max(batch_count, 1)
            print(f"Precily: Epoch [{epoch + 1}/{self.hyperparameters['epochs']}] " f"Training Loss: {epoch_loss:.4f}")
            if self.is_wandb_enabled():
                self.log_metrics({"train_loss": epoch_loss}, step=epoch)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """Predict LN(IC50) for the given cell line / drug pairs.

        :param cell_line_ids: cell line identifiers
        :param drug_ids: drug identifiers
        :param cell_line_input: cell line pathway features
        :param drug_input: drug SMILESVec features

        :returns: predicted response values

        :raises ValueError: if drug_input is None or the model is not built
        """
        if drug_input is None:
            raise ValueError("Precily model requires drug features.")
        if self.model is None:
            raise ValueError("Precily model not initialized.")

        predict_dataset = _PrecilyDataset(
            response=np.zeros(len(cell_line_ids)),
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
        )
        predict_loader = DataLoader(
            predict_dataset,
            batch_size=self.hyperparameters.get("batch_size", 128),
            shuffle=False,
        )

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for pathway_inputs, drug_inputs, _ in predict_loader:
                pathway_inputs = pathway_inputs.to(self.DEVICE)
                drug_inputs = drug_inputs.to(self.DEVICE)
                x = torch.cat([pathway_inputs, drug_inputs], dim=1)
                outputs = self.model(x)
                if outputs.numel() > 1:
                    predictions += outputs.cpu().tolist()
                else:
                    predictions += [outputs.item()]
        return np.asarray(predictions)
