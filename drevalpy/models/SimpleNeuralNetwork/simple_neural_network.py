"""Contains the SimpleNeuralNetwork model."""

import json
import os
import platform
import warnings

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..drp_model import DRPModel
from ..utils import load_and_select_gene_features, load_drug_fingerprint_features, scale_gene_expression
from .utils import FeedForwardNetwork


class SimpleNeuralNetwork(DRPModel):
    """Simple Feedforward Neural Network model with dropout using only gene expression data."""

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]
    early_stopping = True

    def __init__(self):
        """Initializes the SimpleNeuralNetwork.

        The model is built in train(). The gene_expression_scalar is set to the StandardScaler() and later fitted
        using the training data only.
        """
        super().__init__()
        self.model = None
        self.hyperparameters = None
        self.gene_expression_scaler = StandardScaler()

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SimpleNeuralNetwork
        """
        return "SimpleNeuralNetwork"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        :param hyperparameters: includes units_per_layer and dropout_prob.
        """
        self.hyperparameters = hyperparameters
        self.hyperparameters.setdefault("input_dim_gex", None)
        self.hyperparameters.setdefault("input_dim_fp", None)

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        First scales the gene expression data and trains the model.

        The gene expression data is first arcsinh transformed. Afterward, the StandardScaler() is fitted on the
        training gene expression data only. Then, it transforms all gene expression data.
        :param output: training data associated with the response output
        :param cell_line_input: cell line omics features
        :param drug_input: drug omics features
        :param output_earlystopping: optional early stopping dataset
        :param model_checkpoint_dir: directory to save the model checkpoints
        :raises ValueError: if drug_input (fingerprints) is missing

        """
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) are required for SimpleNeuralNetwork.")

        # Apply arcsinh transformation and scaling to gene expression features
        if "gene_expression" in self.cell_line_views:
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(output.cell_line_ids),
                training=True,
                gene_expression_scaler=self.gene_expression_scaler,
            )

        dim_gex = next(iter(cell_line_input.features.values()))["gene_expression"].shape[0]
        dim_fingerprint = next(iter(drug_input.features.values()))["fingerprints"].shape[0]
        self.hyperparameters["input_dim_gex"] = dim_gex
        self.hyperparameters["input_dim_fp"] = dim_fingerprint

        self.model = FeedForwardNetwork(
            hyperparameters=self.hyperparameters,
            input_dim=dim_gex + dim_fingerprint,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*does not have many workers which may be a bottleneck.*",
            )
            warnings.filterwarnings(
                "ignore",
                message="Starting from v1\\.9\\.0, `tensorboardX` has been removed.*",
            )
            if (output_earlystopping is not None) and len(output_earlystopping) == 0:
                output_earlystopping = output
                print("SimpleNeuralNetwork: Early stopping dataset empty. Using training data for early stopping")

                print("Probably, your training dataset is small.")

            self.model.fit(
                output_train=output,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
                cell_line_views=self.cell_line_views,
                drug_views=self.drug_views,
                output_earlystopping=output_earlystopping,
                trainer_params={
                    "max_epochs": self.hyperparameters.get("max_epochs", 100),
                    "progress_bar_refresh_rate": 500,
                },
                batch_size=16,
                patience=5,
                num_workers=1 if platform.system() == "Windows" else 8,
                model_checkpoint_dir=model_checkpoint_dir,
            )

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the response for the given input.

        :param cell_line_ids: IDs of the cell lines to be predicted
        :param drug_ids: IDs of the drugs to be predicted
        :param cell_line_input: gene expression of the test data
        :param drug_input: fingerprints of the test data
        :returns: the predicted drug responses
        """
        # Apply arcsinh transformation and scaling to gene expression features
        if "gene_expression" in self.cell_line_views:
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(cell_line_ids),
                training=False,
                gene_expression_scaler=self.gene_expression_scaler,
            )

        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

        return self.model.predict(x)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: Path to the gene expression and landmark genes
        :param dataset_name: name of the dataset
        :return: FeatureDataset containing the cell line gene expression features, filtered through the landmark genes
        """
        return load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list="landmark_genes",
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the fingerprint data.

        :param data_path: Path to the fingerprints, e.g., data/
        :param dataset_name: name of the dataset, e.g., GDSC1
        :returns: FeatureDataset containing the fingerprints
        """
        return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)

    def save(self, directory: str) -> None:
        """
        Save the trained model, hyperparameters, and gene expression scaler to the given directory.

        This enables full reconstruction of the model using `load`.

        Files saved:
        - model.pt: PyTorch state_dict of the trained model
        - hyperparameters.json: Dictionary containing all relevant model hyperparameters
        - scaler.pkl: Fitted StandardScaler for gene expression features

        :param directory: Target directory to store all model artifacts
        """
        os.makedirs(directory, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(directory, "model.pt"))  # noqa: S614

        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(self.hyperparameters, f)

        joblib.dump(self.gene_expression_scaler, os.path.join(directory, "scaler.pkl"))

    @classmethod
    def load(cls, directory: str) -> "SimpleNeuralNetwork":
        """
        Load a trained SimpleNeuralNetwork instance from disk.

        This includes:
        - model.pt: PyTorch state_dict of the trained model
        - hyperparameters.json: Dictionary with model hyperparameters
        - scaler.pkl: Fitted StandardScaler for gene expression features

        :param directory: Directory containing the saved model files
        :return: An instance of SimpleNeuralNetwork with restored state
        :raises FileNotFoundError: if any required file is missing
        """
        hyperparam_file = os.path.join(directory, "hyperparameters.json")
        scaler_file = os.path.join(directory, "scaler.pkl")
        model_file = os.path.join(directory, "model.pt")

        if not all(os.path.exists(f) for f in [hyperparam_file, scaler_file, model_file]):
            raise FileNotFoundError("Missing model files. Required: model.pt, hyperparameters.json, scaler.pkl")

        instance = cls()

        with open(hyperparam_file) as f:
            instance.hyperparameters = json.load(f)

        instance.gene_expression_scaler = joblib.load(scaler_file)

        dim_gex = instance.hyperparameters["input_dim_gex"]
        dim_fp = instance.hyperparameters["input_dim_fp"]

        instance.model = FeedForwardNetwork(instance.hyperparameters, input_dim=dim_gex + dim_fp)
        instance.model.load_state_dict(torch.load(model_file))  # noqa: S614
        instance.model.eval()

        return instance
