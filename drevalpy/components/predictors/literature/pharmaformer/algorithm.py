"""Contains PharmaFormer, a transformer-based deep learning model for drug response prediction.

A Transformer-based deep learning model designed to predict clinical drug responses
by integrating gene expression profiles and drug molecular structures.

Original authors: Zhou et al. (2025, 10.1038/s41698-025-01082-6)
Code adapted from their Github: https://github.com/zhouyuru1205/PharmaFormer
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from drevalpy.components.predictors.literature._training_helpers import LiteratureTrainingMixin
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from .model_utils import CombinedModel


class _PharmaFormerDataset(Dataset):
    """PyTorch Dataset for PharmaFormer model."""

    def __init__(
        self,
        response: np.ndarray,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_features: FeatureDataset,
        drug_features: FeatureDataset,
    ):
        """Initialize the dataset.

        :param response: Drug response values
        :param cell_line_ids: Cell line identifiers
        :param drug_ids: Drug identifiers
        :param cell_line_features: FeatureDataset with cell line features
        :param drug_features: FeatureDataset with drug features
        """
        self.response = response
        self.cell_line_ids = cell_line_ids
        self.drug_ids = drug_ids
        self.cell_line_features = cell_line_features
        self.drug_features = drug_features

    def __len__(self) -> int:
        """Return the length of the dataset.

        :returns: return: Length of the dataset
        """
        return len(self.response)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset.

        :param idx: Index of the item

        :returns: return: Tuple of (gene_features, drug_features, response)
        """
        cell_line_id = self.cell_line_ids[idx]
        drug_id = self.drug_ids[idx]

        gene_features = torch.tensor(
            self.cell_line_features.features[cell_line_id]["gene_expression"], dtype=torch.float32
        )
        drug_features = torch.tensor(self.drug_features.features[drug_id]["bpe_smiles"], dtype=torch.float32)
        response = torch.tensor(self.response[idx], dtype=torch.float32)

        return gene_features, drug_features, response


class PharmaFormerModel(LiteratureTrainingMixin):
    """PharmaFormer model for drug response prediction."""

    cell_line_views = ["gene_expression"]
    drug_views = ["bpe_smiles"]
    early_stopping = True

    def __init__(self) -> None:
        """Initialize the PharmaFormer model."""
        super().__init__()
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: CombinedModel | None = None
        self.hyperparameters: dict[str, Any] = {}
        self._saved_gene_input_size: int | None = None

    @classmethod
    def get_model_name(cls) -> str:
        """Get the model name.

        :returns: returns: PharmaFormer
        """
        return "PharmaFormer"

    @classmethod
    def get_default_hyperparameters(cls) -> dict[str, Any]:
        """Return default PharmaFormer hyperparameters.

        :returns: Default hyperparameter mapping.
        """
        return {
            "gene_hidden_size": 2048,
            "drug_hidden_size": 128,
            "feature_dim": 64,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 1024,
            "dropout": 0.1,
            "batch_size": 64,
            "lr": 0.00001,
            "epochs": 100,
            "patience": 10,
        }

    def configure(self, hyperparameters: dict[str, Any]) -> None:
        """Builds the PharmaFormer model with the specified hyperparameters.

        :param hyperparameters: Keys include hidden sizes, ``nhead``, ``num_layers``,
            ``dim_feedforward``, ``dropout``, ``batch_size``, ``lr``, ``epochs``, and ``patience``.
        """
        # Log hyperparameters to wandb if enabled
        self.log_hyperparameters(hyperparameters)

        self.hyperparameters = hyperparameters
        # Model will be built in train() when we know the input dimensions

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str | Path = "checkpoints",
    ) -> None:
        """Trains the model.

        :param output: training data associated with the response output
        :param cell_line_input: input data associated with the cell line
        :param drug_input: input data associated with the drug
        :param output_earlystopping: early stopping data associated with the response output
        :param model_checkpoint_dir: directory to save the model checkpoint

        :raises ValueError: if drug_input is None or if early stopping data is missing
        """
        if drug_input is None:
            raise ValueError("PharmaFormer model requires drug features.")
        if output_earlystopping is None:
            raise ValueError("PharmaFormer model requires early stopping data.")

        from .pharmaformer_training import run_pharmaformer_training

        run_pharmaformer_training(
            engine=self,
            output=output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
            output_earlystopping=output_earlystopping,
            model_checkpoint_dir=model_checkpoint_dir,
            pharmaformer_dataset_cls=_PharmaFormerDataset,
        )

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """Predicts the response values for the given cell lines and drugs.

        :param cell_line_ids: list of cell line IDs
        :param drug_ids: list of drug IDs
        :param cell_line_input: input data associated with the cell line
        :param drug_input: input data associated with the drug

        :returns: return: predicted response values

        :raises ValueError: if drug_input is None or if the model is not initialized
        """
        if drug_input is None:
            raise ValueError("PharmaFormer model requires drug features.")
        if self.model is None:
            raise ValueError("PharmaFormer model not initialized.")

        # Create dataset
        predict_dataset = _PharmaFormerDataset(
            response=np.zeros(len(cell_line_ids)),
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
        )

        predict_loader = DataLoader(
            predict_dataset, batch_size=self.hyperparameters.get("batch_size", 64), shuffle=False
        )

        # Run prediction
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for gene_inputs, smiles_inputs, _ in predict_loader:
                gene_inputs = gene_inputs.to(self.DEVICE)
                smiles_inputs = smiles_inputs.to(self.DEVICE)

                outputs = self.model(gene_inputs, smiles_inputs)
                if outputs.numel() > 1:
                    predictions += outputs.squeeze().cpu().tolist()
                else:
                    predictions += [outputs.item()]
        return np.asarray(predictions)
