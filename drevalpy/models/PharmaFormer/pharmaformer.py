"""
Contains PharmaFormer, a transformer-based deep learning model for drug response prediction.

A Transformer-based deep learning model designed to predict clinical drug responses
by integrating gene expression profiles and drug molecular structures.

Original authors: Zhou et al. (2025, 10.1038/s41698-025-01082-6)
Code adapted from their Github: https://github.com/zhouyuru1205/PharmaFormer
"""

import json
import os
import secrets
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.utils import load_and_select_gene_features

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
        """
        Initialize the dataset.

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

        :return: Length of the dataset
        """
        return len(self.response)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single item from the dataset.

        :param idx: Index of the item
        :return: Tuple of (gene_features, drug_features, response)
        """
        cell_line_id = self.cell_line_ids[idx]
        drug_id = self.drug_ids[idx]

        gene_features = torch.tensor(
            self.cell_line_features.features[cell_line_id]["gene_expression"], dtype=torch.float32
        )
        drug_features = torch.tensor(self.drug_features.features[drug_id]["bpe_smiles"], dtype=torch.float32)
        response = torch.tensor(self.response[idx], dtype=torch.float32)

        return gene_features, drug_features, response


class PharmaFormerModel(DRPModel):
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
        self.gene_expression_scaler: StandardScaler | None = None
        self.gene_expression_normalizer: MinMaxScaler | None = None

    @classmethod
    def get_model_name(cls) -> str:
        """
        Get the model name.

        :returns: PharmaFormer
        """
        return "PharmaFormer"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Builds the PharmaFormer model with the specified hyperparameters.

        :param hyperparameters: Model hyperparameters including gene_hidden_size, drug_hidden_size,
            feature_dim, nhead, num_layers, dim_feedforward, dropout, batch_size, lr, epochs, patience
        """
        self.hyperparameters = hyperparameters
        # Model will be built in train() when we know the input dimensions

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Trains the model.

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

        # Get feature dimensions
        train_gene_features = cell_line_input.get_feature_matrix(
            view="gene_expression", identifiers=output.cell_line_ids
        )
        gene_input_size = train_gene_features.shape[1]

        # Standardize and normalize gene expression (matching original PharmaFormer)
        self.gene_expression_scaler = StandardScaler()
        self.gene_expression_normalizer = MinMaxScaler()

        train_gene_scaled = self.gene_expression_scaler.fit_transform(train_gene_features)
        self.gene_expression_normalizer.fit_transform(train_gene_scaled)

        # Apply transformations to all gene expression features
        cell_line_input = cell_line_input.copy()
        for cell_line_id in cell_line_input.features:
            gene_expr = cell_line_input.features[cell_line_id]["gene_expression"]
            gene_expr_scaled = self.gene_expression_scaler.transform(gene_expr.reshape(1, -1))
            gene_expr_normalized = self.gene_expression_normalizer.transform(gene_expr_scaled)
            cell_line_input.features[cell_line_id]["gene_expression"] = gene_expr_normalized.flatten()

        # Build model with known input dimensions
        self.model = CombinedModel(
            gene_input_size=gene_input_size,
            gene_hidden_size=self.hyperparameters["gene_hidden_size"],
            drug_hidden_size=self.hyperparameters["drug_hidden_size"],
            feature_dim=self.hyperparameters["feature_dim"],
            nhead=self.hyperparameters["nhead"],
            num_layers=self.hyperparameters.get("num_layers", 3),
            dim_feedforward=self.hyperparameters.get("dim_feedforward", 2048),
            dropout=self.hyperparameters.get("dropout", 0.1),
        ).to(self.DEVICE)

        loss_func = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters["lr"])

        # Create datasets
        train_dataset = _PharmaFormerDataset(
            response=output.response,
            cell_line_ids=output.cell_line_ids,
            drug_ids=output.drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
        )
        early_stopping_dataset = _PharmaFormerDataset(
            response=output_earlystopping.response,
            cell_line_ids=output_earlystopping.cell_line_ids,
            drug_ids=output_earlystopping.drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.hyperparameters["batch_size"],
            shuffle=True,
        )
        early_stopping_loader = DataLoader(
            early_stopping_dataset,
            batch_size=self.hyperparameters["batch_size"],
            shuffle=False,
        )

        # Early stopping parameters
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        # Ensure the checkpoint directory exists
        os.makedirs(model_checkpoint_dir, exist_ok=True)
        version = "version-" + "".join([secrets.choice("0123456789abcdef") for _ in range(20)])

        checkpoint_path = os.path.join(model_checkpoint_dir, f"{version}_best_PharmaFormer_model.pth")

        # Train model
        print("Training PharmaFormer model")
        for epoch in range(self.hyperparameters["epochs"]):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

            # Training phase
            for gene_inputs, smiles_inputs, targets in train_loader:
                gene_inputs = gene_inputs.to(self.DEVICE)
                smiles_inputs = smiles_inputs.to(self.DEVICE)
                targets = targets.to(self.DEVICE)

                # Forward pass
                outputs = self.model(gene_inputs, smiles_inputs)
                loss = loss_func(outputs.squeeze(), targets)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().item()
                batch_count += 1

            epoch_loss /= batch_count
            print(f"PharmaFormer: Epoch [{epoch + 1}/{self.hyperparameters['epochs']}] Training Loss: {epoch_loss:.4f}")

            # Validation phase for early stopping
            self.model.eval()
            val_loss = 0.0
            val_batch_count = 0
            with torch.no_grad():
                for gene_inputs, smiles_inputs, targets in early_stopping_loader:
                    gene_inputs = gene_inputs.to(self.DEVICE)
                    smiles_inputs = smiles_inputs.to(self.DEVICE)
                    targets = targets.to(self.DEVICE)

                    outputs = self.model(gene_inputs, smiles_inputs)
                    loss = loss_func(outputs.squeeze(), targets)

                    val_loss += loss.item()
                    val_batch_count += 1

            val_loss /= val_batch_count
            print(f"PharmaFormer: Epoch [{epoch + 1}/{self.hyperparameters['epochs']}] Validation Loss: {val_loss:.4f}")

            # Checkpointing: Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), checkpoint_path)  # noqa: S614
                print(f"PharmaFormer: Saved best model at epoch {epoch + 1}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.hyperparameters.get("patience", 10):
                    print(f"PharmaFormer: Early stopping triggered at epoch {epoch + 1}")
                    break

        # Reload the best model after training
        print("PharmaFormer: Reloading the best model")
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.DEVICE, weights_only=True)
        )  # noqa: S614
        self.model.to(self.DEVICE)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the response values for the given cell lines and drugs.

        :param cell_line_ids: list of cell line IDs
        :param drug_ids: list of drug IDs
        :param cell_line_input: input data associated with the cell line
        :param drug_input: input data associated with the drug
        :return: predicted response values
        :raises ValueError: if drug_input is None or if the model is not initialized
        """
        if drug_input is None:
            raise ValueError("PharmaFormer model requires drug features.")
        if self.model is None:
            raise ValueError("PharmaFormer model not initialized.")

        # Apply transformations to gene expression if scalers are available
        if self.gene_expression_scaler is not None and self.gene_expression_normalizer is not None:
            cell_line_input = cell_line_input.copy()
            for cell_line_id in cell_line_ids:
                if cell_line_id in cell_line_input.features:
                    gene_expr = cell_line_input.features[cell_line_id]["gene_expression"]
                    gene_expr_scaled = self.gene_expression_scaler.transform(gene_expr.reshape(1, -1))
                    gene_expr_normalized = self.gene_expression_normalizer.transform(gene_expr_scaled)
                    cell_line_input.features[cell_line_id]["gene_expression"] = gene_expr_normalized.flatten()

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

        return np.array(predictions)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load cell line features.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: cell line features
        """
        return load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list="landmark_genes_reduced",
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load drug features (BPE-encoded SMILES).

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: drug features
        :raises FileNotFoundError: if the BPE SMILES file is not found
        """
        bpe_smiles_file = os.path.join(data_path, dataset_name, "drug_bpe_smiles.csv")
        if not os.path.exists(bpe_smiles_file):
            raise FileNotFoundError(
                f"BPE SMILES file not found: {bpe_smiles_file}. "
                "Please run the BPE featurizer first: "
                "python -m drevalpy.datasets.featurizer.create_bpe_smiles_embeddings <dataset_name>"
            )

        bpe_df = pd.read_csv(bpe_smiles_file, dtype={"pubchem_id": str})
        features = {}
        for _, row in bpe_df.iterrows():
            drug_id = row["pubchem_id"]
            # Extract all feature columns (excluding pubchem_id)
            embedding = row.drop("pubchem_id").values.astype(np.float32)
            features[drug_id] = {"bpe_smiles": embedding}

        return FeatureDataset(features)

    def save(self, directory: str) -> None:
        """
        Save the PharmaFormer model using PyTorch conventions.

        This method stores:
        - "pharmaformer_model.pt": PyTorch state_dict of the model
        - "hyperparameters.json": All hyperparameters
        - "gene_scaler.pkl": Fitted StandardScaler for gene expression
        - "gene_normalizer.pkl": Fitted MinMaxScaler for gene expression

        :param directory: Target directory where the model files will be saved
        :raises ValueError: If model is not built
        """
        import joblib

        os.makedirs(directory, exist_ok=True)
        if self.model is None:
            raise ValueError("Cannot save model: model is not built.")

        model = cast(CombinedModel, self.model)

        torch.save(model.state_dict(), os.path.join(directory, "pharmaformer_model.pt"))  # noqa: S614

        # Save hyperparameters including gene_input_size
        save_hyperparameters = self.hyperparameters.copy()
        if self.model is not None:
            # Extract gene_input_size from the model
            save_hyperparameters["gene_input_size"] = self.model.feature_extractor.gene_fc1.in_features

        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(save_hyperparameters, f)

        if self.gene_expression_scaler is not None:
            joblib.dump(self.gene_expression_scaler, os.path.join(directory, "gene_scaler.pkl"))
        if self.gene_expression_normalizer is not None:
            joblib.dump(self.gene_expression_normalizer, os.path.join(directory, "gene_normalizer.pkl"))

    @classmethod
    def load(cls, directory: str) -> "PharmaFormerModel":
        """
        Load the PharmaFormer model using PyTorch conventions.

        This method expects the following files in the given directory:
        - "pharmaformer_model.pt": PyTorch state_dict of the model
        - "hyperparameters.json": Dictionary of hyperparameters
        - "gene_scaler.pkl": Fitted StandardScaler (optional)
        - "gene_normalizer.pkl": Fitted MinMaxScaler (optional)

        :param directory: Path to the directory containing the model files
        :return: An instance of PharmaFormerModel with loaded model
        """
        import joblib

        instance = cls()

        with open(os.path.join(directory, "hyperparameters.json")) as f:
            instance.hyperparameters = json.load(f)

        # Load scalers if they exist
        scaler_path = os.path.join(directory, "gene_scaler.pkl")
        normalizer_path = os.path.join(directory, "gene_normalizer.pkl")
        if os.path.exists(scaler_path):
            instance.gene_expression_scaler = joblib.load(scaler_path)
        if os.path.exists(normalizer_path):
            instance.gene_expression_normalizer = joblib.load(normalizer_path)

        # Model will be built when needed (requires input dimensions)
        # For now, we'll need to rebuild it with the saved hyperparameters
        # This requires knowing the gene_input_size, which should be saved in hyperparameters
        if "gene_input_size" in instance.hyperparameters:
            instance.model = CombinedModel(
                gene_input_size=instance.hyperparameters["gene_input_size"],
                gene_hidden_size=instance.hyperparameters["gene_hidden_size"],
                drug_hidden_size=instance.hyperparameters["drug_hidden_size"],
                feature_dim=instance.hyperparameters["feature_dim"],
                nhead=instance.hyperparameters["nhead"],
                num_layers=instance.hyperparameters.get("num_layers", 3),
                dim_feedforward=instance.hyperparameters.get("dim_feedforward", 2048),
                dropout=instance.hyperparameters.get("dropout", 0.1),
            ).to(instance.DEVICE)

            instance.model.load_state_dict(
                torch.load(os.path.join(directory, "pharmaformer_model.pt"), map_location=instance.DEVICE)  # noqa: S614
            )
            instance.model.eval()

        return instance
