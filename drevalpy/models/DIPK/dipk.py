"""
DIPK model. Adapted from https://github.com/user15632/DIPK.

Original publication:
Improving drug response prediction via integrating gene relationships with deep learning
Pengyong Li, Zhengxiang Jiang, Tianxiao Liu, Xinyu Liu, Hui Qiao, Xiaojun Yao
Briefings in Bioinformatics, Volume 25, Issue 3, May 2024, bbae153, https://doi.org/10.1093/bib/bbae153
"""

import json
import os
import secrets
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.utils import load_and_select_gene_features

from .data_utils import CollateFn, DIPKDataset, get_data, load_bionic_features
from .gene_expression_encoder import GeneExpressionEncoder, encode_gene_expression, train_gene_expession_autoencoder
from .model_utils import Predictor


class DIPKModel(DRPModel):
    """DIPK model. Adapted from https://github.com/user15632/DIPK."""

    cell_line_views = ["gene_expression", "bionic_features"]
    drug_views = ["molgnet_features"]
    early_stopping = True

    def __init__(self) -> None:
        """Initialize the DIPK model."""
        super().__init__()
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # all of this gets initialized in build_model
        self.model: Predictor | None = None
        self.gene_expression_encoder: GeneExpressionEncoder | None = None
        self.hyperparameters: dict[str, Any] = {}

    @classmethod
    def get_model_name(cls) -> str:
        """
        Get the model name.

        :returns: DIPK
        """
        return "DIPK"

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Builds the DIPK model with the specified hyperparameters.

        :param hyperparameters: embedding_dim, heads, fc_layer_num, fc_layer_dim, dropout_rate, epochs, batch_size, lr

        Details of hyperparameters:

        - embedding_dim: int, embedding dimension used for the graph encoder which is not used in the final model
        - heads: int, number of heads for the multi-head attention layer, defaults to 1
        - fc_layer_num: int, number of fully connected layers for the dense layers
        - fc_layer_dim: list[int], number of neurons for each fully connected layer
        - dropout_rate: float, dropout rate for all fully connected layers
        - epochs: int, number of epochs to train the model
        - batch_size: int, batch size for training
        - lr: float, learning rate for training
        """
        self.model = Predictor(
            hyperparameters["heads"],
            hyperparameters["fc_layer_num"],
            hyperparameters["fc_layer_dim"],
            hyperparameters["dropout_rate"],
        ).to(self.DEVICE)
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
        Trains the model.

        :param output: training data associated with the response output
        :param cell_line_input: input data associated with the cell line
        :param drug_input: input data associated with the drug
        :param output_earlystopping: early stopping data associated with the response output
        :param model_checkpoint_dir: directory to save the model checkpoint
        :raises ValueError: if drug_input is None or if the model is not initialized
        """
        if drug_input is None:
            raise ValueError("DIPK model requires drug features.")
        if not isinstance(self.model, Predictor):
            raise ValueError("DIPK model not initialized.")
        if output_earlystopping is None:
            raise ValueError("DIPK model requires early stopping data.")

        loss_func = nn.MSELoss()
        params = [{"params": self.model.parameters()}]
        optimizer = optim.Adam(params, lr=self.hyperparameters["lr"])

        train_gene_expression = cell_line_input.get_feature_matrix(
            view="gene_expression", identifiers=output.cell_line_ids
        )
        val_gene_expression = cell_line_input.get_feature_matrix(
            view="gene_expression", identifiers=output_earlystopping.cell_line_ids
        )

        self.gene_expression_encoder = train_gene_expession_autoencoder(
            train_gene_expression,
            val_gene_expression,
            epochs_autoencoder=self.hyperparameters["epochs_autoencoder"],
        )
        self.hyperparameters["gene_encoder_input_dim"] = train_gene_expression.shape[1]

        cell_line_input.apply(
            lambda x: encode_gene_expression(x, self.gene_expression_encoder),  # type: ignore[arg-type]
            view="gene_expression",
        )  # type: ignore[arg-type]

        # Load data
        collate = CollateFn(train=True)
        train_samples = get_data(
            cell_ids=output.cell_line_ids,
            drug_ids=output.drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
            ic50=output.response,
        )
        early_stopping_samples = get_data(
            cell_ids=output_earlystopping.cell_line_ids,
            drug_ids=output_earlystopping.drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
            ic50=output_earlystopping.response,
        )

        train_loader: DataLoader = DataLoader(
            DIPKDataset(train_samples), batch_size=self.hyperparameters["batch_size"], shuffle=True, collate_fn=collate
        )
        early_stopping_loader: DataLoader = DataLoader(
            DIPKDataset(early_stopping_samples),
            batch_size=self.hyperparameters["batch_size"],
            shuffle=True,
            collate_fn=collate,
        )

        # Early stopping parameters
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        # Ensure the checkpoint directory exists
        os.makedirs(model_checkpoint_dir, exist_ok=True)
        version = "version-" + "".join(
            [secrets.choice("0123456789abcdef") for _ in range(20)]
        )  # preventing conflicts of filenames

        checkpoint_path = os.path.join(model_checkpoint_dir, f"{version}_best_DIPK_model.pth")

        # Train model
        print("Training DIPK model")
        for epoch in range(self.hyperparameters["epochs"]):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

            # Training phase
            for batch in train_loader:
                drug_features = batch["molgnet_features"].to(self.DEVICE)
                gene_features = batch["gene_features"].to(self.DEVICE)
                bionic_features = batch["bionic_features"].to(self.DEVICE)
                molgnet_mask = batch["molgnet_mask"].to(self.DEVICE)
                ic50_values = batch["ic50_values"].to(self.DEVICE)

                # Forward pass
                prediction = self.model(
                    molgnet_drug_features=drug_features,
                    gene_expression=gene_features,
                    bionic=bionic_features,
                    molgnet_mask=molgnet_mask,
                )

                # Compute the loss
                loss = loss_func(torch.squeeze(prediction), torch.squeeze(ic50_values))

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update loss and batch count
                epoch_loss += loss.detach().item()
                batch_count += 1

            epoch_loss /= batch_count
            print(f"DIPK: Epoch [{epoch + 1}] Training Loss: {epoch_loss:.4f}")

            # Validation phase for early stopping
            self.model.eval()
            val_loss = 0.0
            val_batch_count = 0
            with torch.no_grad():
                for batch in early_stopping_loader:
                    drug_features = batch["molgnet_features"].to(self.DEVICE)
                    gene_features = batch["gene_features"].to(self.DEVICE)
                    bionic_features = batch["bionic_features"].to(self.DEVICE)
                    molgnet_mask = batch["molgnet_mask"].to(self.DEVICE)
                    ic50_values = batch["ic50_values"].to(self.DEVICE)

                    # Forward pass
                    prediction = self.model(
                        molgnet_drug_features=drug_features,
                        gene_expression=gene_features,
                        bionic=bionic_features,
                        molgnet_mask=molgnet_mask,
                    )

                    # Compute the loss
                    loss = loss_func(torch.squeeze(prediction), torch.squeeze(ic50_values))

                    # Update validation loss
                    val_loss += loss.item()
                    val_batch_count += 1

            val_loss /= val_batch_count
            print(f"DIPK: Epoch [{epoch + 1}] Validation Loss: {val_loss:.4f}")

            # Checkpointing: Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save the model checkpoint securely
                torch.save(self.model.state_dict(), checkpoint_path)  # noqa S614
                print(f"DIPK: Saved best model at epoch {epoch + 1}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.hyperparameters["patience"]:
                    print(f"DIPK: Early stopping triggered at epoch {epoch + 1}")
                    break

        # Reload the best model after training
        print("DIPK: Reloading the best model")
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.DEVICE, weights_only=True)  # noqa S614
        )
        self.model.to(self.DEVICE)  # Ensure model is on the correct device

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
        :raises ValueError: if drug_input is None or if the model is not initialized or
            if the gene expression encoder is not initialized
        """
        if drug_input is None:
            raise ValueError("DIPK model requires drug features.")
        if not isinstance(self.model, Predictor):
            raise ValueError("DIPK model not initialized.")

        # Encode gene expression data if this has not been done yet (e.g., for cross-study predictions)
        if self.gene_expression_encoder is None:
            raise ValueError("Gene expression encoder is not initialized.")
        random_cell_line = next(iter(cell_line_input.features.keys()))
        if (
            len(cell_line_input.features[random_cell_line]["gene_expression"])
            != self.gene_expression_encoder.latent_dim
        ):
            print("Encoding gene expression data for cross study prediction")
            cell_line_input.apply(
                lambda x: encode_gene_expression(x, self.gene_expression_encoder),  # type: ignore[arg-type]
                view="gene_expression",
            )  # type: ignore[arg-type]

        # Load data
        collate = CollateFn(train=False)
        test_samples = get_data(
            cell_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_features=cell_line_input,
            drug_features=drug_input,
        )
        test_loader: DataLoader = DataLoader(
            DIPKDataset(test_samples), batch_size=self.hyperparameters["batch_size"], shuffle=False, collate_fn=collate
        )

        # Run prediction
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                drug_features = batch["molgnet_features"].to(self.DEVICE)
                gene_features = batch["gene_features"].to(self.DEVICE)
                bionic_features = batch["bionic_features"].to(self.DEVICE)
                molgnet_mask = batch["molgnet_mask"].to(self.DEVICE)

                prediction = self.model(
                    molgnet_drug_features=drug_features,
                    gene_expression=gene_features,
                    bionic=bionic_features,
                    molgnet_mask=molgnet_mask,
                )
                if prediction.numel() > 1:
                    predictions += torch.squeeze(prediction).cpu().tolist()
                else:
                    predictions += [prediction.item()]
        return np.array(predictions)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load cell line features.

        :param data_path: path to the data
        :param dataset_name: path to the dataset
        :returns: cell line features
        """
        # we use the interception of all genes that are present
        # in the gene expression features of all datasets
        gene_expression = load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list="gene_expression_intersection",
            data_path=data_path,
            dataset_name=dataset_name,
        )
        bionic_features = load_bionic_features(
            data_path=data_path,
            dataset_name=dataset_name,
        )
        bionic_features.add_features(gene_expression)

        return bionic_features

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load drug features.

        :param data_path: path to the data
        :param dataset_name: path to the dataset
        :returns: drug features
        """

        def load_feature(file_path, sep="\t"):
            return np.array(pd.read_csv(file_path, index_col=0, sep=sep))

        drug_path = os.path.join(data_path, dataset_name, "DIPK_features", "Drugs")
        files_in_drug_path = os.listdir(drug_path)
        drug_list = [
            file.split("_")[1].split(".csv")[0]
            for file in files_in_drug_path
            if file.endswith(".csv") and file.startswith("MolGNet")
        ]

        f = FeatureDataset(
            features={
                drug: {
                    "molgnet_features": load_feature(os.path.join(drug_path, f"MolGNet_{drug}.csv")),
                }
                for drug in drug_list
            }
        )

        return f

    def save(self, directory: str) -> None:
        """
        Save the DIPK model and gene expression encoder using PyTorch conventions.

        This method stores:
        - "dipk_model.pt": PyTorch state_dict of the DIPK predictor model
        - "gene_encoder.pt": PyTorch state_dict of the trained gene expression encoder
        - "hyperparameters.json": All hyperparameters including encoder input_dim

        :param directory: Target directory where the model files will be saved
        :raises ValueError: If model or encoder is not built
        """
        os.makedirs(directory, exist_ok=True)
        if self.model is None or self.gene_expression_encoder is None:
            raise ValueError("Cannot save model: model is not built.")
        model = cast(Predictor, self.model)

        torch.save(model.state_dict(), os.path.join(directory, "dipk_model.pt"))  # noqa: S614
        torch.save(self.gene_expression_encoder.state_dict(), os.path.join(directory, "gene_encoder.pt"))  # noqa: S614
        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(self.hyperparameters, f)

    @classmethod
    def load(cls, directory: str) -> "DIPKModel":
        """
        Load the DIPK model and gene expression encoder using PyTorch conventions.

        This method expects the following files in the given directory:
        - "dipk_model.pt": PyTorch state_dict of the DIPK predictor model
        - "gene_encoder.pt": PyTorch state_dict of the gene expression encoder
        - "hyperparameters.json": Dictionary of hyperparameters, must include "gene_encoder_input_dim"

        :param directory: Path to the directory containing the model files
        :return: An instance of DIPK with loaded model and encoder
        """
        instance = cls()

        with open(os.path.join(directory, "hyperparameters.json")) as f:
            instance.hyperparameters = json.load(f)

        instance.build_model(instance.hyperparameters)
        instance.model = cast(Predictor, instance.model)

        instance.model.load_state_dict(
            torch.load(os.path.join(directory, "dipk_model.pt"), map_location=instance.DEVICE)  # noqa: S614
        )
        instance.model.eval()

        input_dim = instance.hyperparameters["gene_encoder_input_dim"]
        instance.gene_expression_encoder = GeneExpressionEncoder(input_dim=input_dim)
        instance.gene_expression_encoder.load_state_dict(
            torch.load(os.path.join(directory, "gene_encoder.pt"), map_location=instance.DEVICE)  # noqa: S614
        )
        instance.gene_expression_encoder.eval()

        return instance
