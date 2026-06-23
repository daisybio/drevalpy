r"""
Precily model for drug response prediction.

Contains Precily, a pathway-based deep learning model for drug response
prediction. A deep neural network that predicts LN(IC50) by combining
GSVA pathway-activity scores with SMILESVec drug embeddings.

Original authors: Chawla et al. (2022, 10.1038/s41467-022-33291-z)
Reference code: https://github.com/SmritiChawla/Precily

"""

import json
import os
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel

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
        """
        Initialize the dataset.

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
        """
        Return the number of samples.

        :return: Number of samples in the dataset.
        """
        return len(self.response)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample by index.

        :param idx: sample index
        :return: (pathway_features, drug_features, response) tensors
        """
        cell_line_id = self.cell_line_ids[idx]
        drug_id = self.drug_ids[idx]

        pathway = torch.tensor(self.cell_line_features.features[cell_line_id]["pathways"], dtype=torch.float32)
        drug = torch.tensor(self.drug_features.features[drug_id]["smilesvec"], dtype=torch.float32)
        response = torch.tensor(self.response[idx], dtype=torch.float32)

        return pathway, drug, response


class PrecilyModel(DRPModel):
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
        """
        Get the model name.

        :returns: Precily
        """
        return "Precily"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Store hyperparameters.

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
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Train the Precily model.

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
        """
        Predict LN(IC50) for the given cell line / drug pairs.

        :param cell_line_ids: cell line identifiers
        :param drug_ids: drug identifiers
        :param cell_line_input: cell line pathway features
        :param drug_input: drug SMILESVec features
        :return: predicted response values
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

        return np.array(predictions)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        r"""
        Load precomputed GSVA pathway scores.

        Generate it with the Precily pathway featurizer:
            python -m drevalpy.datasets.featurizer.create_precily_pathway_features <dataset_name> \\
                --gene_sets <path_to/c2.cp.v6.1.symbols.gmt>

        :param data_path: path to the data
        :param dataset_name: dataset name
        :returns: cell line FeatureDataset with the "pathways" view
        :raises FileNotFoundError: if the pathway feature CSV is missing
        """
        pathway_file = os.path.join(data_path, dataset_name, "pathway_features.csv")
        if not os.path.exists(pathway_file):
            raise FileNotFoundError(
                f"Pathway feature file not found: {pathway_file}. "
                "Run the featurizer first: "
                "python -m drevalpy.datasets.featurizer.create_precily_pathway_features "
                f"{dataset_name} --gene_sets <path_to_gmt>"
            )

        df = pd.read_csv(pathway_file, index_col=0)
        features = {cell_line_id: {"pathways": row.values.astype(np.float32)} for cell_line_id, row in df.iterrows()}
        return FeatureDataset(features)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        r"""
        Load precomputed SMILESVec drug embeddings.

        Generate it with the Precily drug featurizer:
            python -m drevalpy.datasets.featurizer.create_precily_drug_embeddings <dataset_name> \\
                --smilesvec_model <path_to/drug.l8.pubchem.canon.ws20.txt>

        :param data_path: path to the data
        :param dataset_name: dataset name
        :returns: drug FeatureDataset with the "smilesvec" view
        :raises FileNotFoundError: if the drug feature CSV is missing
        """
        drug_file = os.path.join(data_path, dataset_name, "drug_smilesvec.csv")
        if not os.path.exists(drug_file):
            raise FileNotFoundError(
                f"Drug SMILESVec feature file not found: {drug_file}. "
                "Run the featurizer first: "
                "python -m drevalpy.datasets.featurizer.create_precily_drug_embeddings "
                f"{dataset_name} --smilesvec_model <path_to_pretrained_model>"
            )

        df = pd.read_csv(drug_file, dtype={"pubchem_id": str})
        features = {}
        for _, row in df.iterrows():
            drug_id = row["pubchem_id"]
            embedding = row.drop("pubchem_id").values.astype(np.float32)
            features[drug_id] = {"smilesvec": embedding}

        return FeatureDataset(features)

    def save(self, directory: str) -> None:
        """
        Save the Precily model using PyTorch conventions.

        Stores:

        - "precily_model.pt": PyTorch state_dict of the network
        - "hyperparameters.json": all hyperparameters plus the resolved
          input_dim (so the network can be rebuilt with the right shape)

        :param directory: target directory
        :raises ValueError: if the model is not built
        """
        os.makedirs(directory, exist_ok=True)
        if self.model is None:
            raise ValueError("Cannot save model: model is not built.")

        model = cast(PrecilyNetwork, self.model)
        torch.save(model.state_dict(), os.path.join(directory, "precily_model.pt"))

        save_hyperparameters = self.hyperparameters.copy()
        save_hyperparameters["input_dim"] = model.net[0].in_features
        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(save_hyperparameters, f)

    @classmethod
    def load(cls, directory: str) -> "PrecilyModel":
        """
        Load a Precily model saved with save method.

        Expects in ``directory``:

        - "precily_model.pt": network state_dict
        - "hyperparameters.json": hyperparameters incl. "input_dim"

        :param directory: directory containing the saved files
        :return: a restored PrecilyModel
        """
        instance = cls()

        with open(os.path.join(directory, "hyperparameters.json")) as f:
            instance.hyperparameters = json.load(f)

        if "input_dim" in instance.hyperparameters:
            instance.model = PrecilyNetwork(
                input_dim=instance.hyperparameters["input_dim"],
                dropout=instance.hyperparameters.get("dropout", 0.1),
            ).to(instance.DEVICE)
            instance.model.load_state_dict(
                torch.load(
                    os.path.join(directory, "precily_model.pt"),
                    map_location=instance.DEVICE,
                    weights_only=True,
                )
            )
            instance.model.eval()

        return instance
