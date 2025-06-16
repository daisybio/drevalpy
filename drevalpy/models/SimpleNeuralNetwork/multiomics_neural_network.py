"""Contains the baseline MultiOmicsNeuralNetwork model."""

import json
import os
import warnings

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..drp_model import DRPModel
from ..utils import get_multiomics_feature_dataset, load_drug_fingerprint_features, prepare_expression_and_methylation
from .utils import FeedForwardNetwork


class MultiOmicsNeuralNetwork(DRPModel):
    """Simple Feedforward Neural Network model with dropout using multiple omics data."""

    cell_line_views = [
        "gene_expression",
        "methylation",
        "mutations",
        "copy_number_variation_gistic",
    ]
    drug_views = ["fingerprints"]
    early_stopping = True

    def __init__(self):
        """
        Initalization method for MultiOmicsNeuralNetwork Model.

        The PCA is initialized to None because it depends on hyperparameter, therefore built in build_model.
        """
        super().__init__()
        self.model = None
        self.hyperparameters = None
        self.methylation_scaler = StandardScaler()
        self.methylation_pca = None
        self.gene_expression_scaler = StandardScaler()

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: MultiOmicsNeuralNetwork
        """
        return "MultiOmicsNeuralNetwork"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        The model is a simple feedforward neural network with dropout. The PCA is used to reduce the dimensionality of
        the methylation data.

        :param hyperparameters: dictionary containing the hyperparameters units_per_layer, dropout_prob, and
            methylation_pca_components.
        """
        self.hyperparameters = hyperparameters
        self.methylation_pca = PCA(n_components=hyperparameters["methylation_pca_components"])

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "",
    ):
        """
        Fits the PCA and trains the model.

        :param output: training data associated with the response output
        :param cell_line_input: cell line omics features
        :param drug_input: drug omics features
        :param output_earlystopping: optional early stopping dataset
        :param model_checkpoint_dir: directory to save the model checkpoints
        :raises ValueError: if drug_input (fingerprints) is missing
        """
        if drug_input is None:
            raise ValueError("Drug input (fingerprints) is needed for the MultiOmicsNeuralNetwork model.")
        cell_line_input = prepare_expression_and_methylation(
            cell_line_input=cell_line_input,
            cell_line_ids=np.unique(output.cell_line_ids),
            training=True,
            gene_expression_scaler=self.gene_expression_scaler,
            methylation_scaler=self.methylation_scaler,
            methylation_pca=self.methylation_pca,
        )

        first_feature = next(iter(cell_line_input.features.values()))
        dim_gex = first_feature["gene_expression"].shape[0]
        dim_met = self.methylation_pca.n_components
        dim_mut = first_feature["mutations"].shape[0]
        dim_cnv = first_feature["copy_number_variation_gistic"].shape[0]
        dim_fingerprint = next(iter(drug_input.features.values()))["fingerprints"].shape[0]

        self.dim_gex = dim_gex
        self.dim_met = dim_met
        self.dim_mut = dim_mut
        self.dim_cnv = dim_cnv
        self.dim_fp = dim_fingerprint

        self.model = FeedForwardNetwork(
            hyperparameters=self.hyperparameters,
            input_dim=dim_gex + dim_met + dim_mut + dim_cnv + dim_fingerprint,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*does not have many workers which may be a bottleneck.*",
            )
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
                num_workers=1,
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
        Applies arcsinh + scaling to gene expression and scaling + PCA to methylation, then predicts.

        :param drug_ids: drug identifiers
        :param cell_line_ids: cell line identifiers
        :param drug_input: drug omics features
        :param cell_line_input: cell line omics features
        :returns: predicted response
        """
        cell_line_input = prepare_expression_and_methylation(
            cell_line_input=cell_line_input,
            cell_line_ids=np.unique(cell_line_ids),
            training=False,
            gene_expression_scaler=self.gene_expression_scaler,
            methylation_scaler=self.methylation_scaler,
            methylation_pca=self.methylation_pca,
        )

        inputs = self.get_feature_matrices(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

        x = np.concatenate(
            (
                inputs["gene_expression"],
                inputs["methylation"],
                inputs["mutations"],
                inputs["copy_number_variation_gistic"],
                inputs["fingerprints"],
            ),
            axis=1,
        )
        return self.model.predict(x)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: data path e.g. data/
        :param dataset_name: dataset name e.g. GDSC1
        :return: FeatureDataset containing the cell line omics features, filtered through the
            drug target genes
        """
        gene_lists = {
            "gene_expression": "drug_target_genes_all_drugs",
            "methylation": "methylation_intersection",
            "mutations": "drug_target_genes_all_drugs",
            "copy_number_variation_gistic": "drug_target_genes_all_drugs",
            "proteomics": "drug_target_genes_all_drugs_proteomics",
        }
        return get_multiomics_feature_dataset(data_path=data_path, gene_lists=gene_lists, dataset_name=dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load the drug features.

        :param data_path: path to the drug features, in this case the drug fingerprints, e.g., data/
        :param dataset_name: name of the dataset, e.g., GDSC1
        :returns: FeatureDataset containing the drug fingerprint features
        """
        return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)

    def save(self, directory: str) -> None:
        """
        Save the trained model, hyperparameters, scalers, PCA object, and feature dimensions to disk.

        Files saved:
        - model.pt
        - hyperparameters.json
        - gene_scaler.pkl
        - methylation_scaler.pkl
        - methylation_pca.pkl
        - metadata.json

        :param directory: Target directory
        """
        os.makedirs(directory, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(directory, "model.pt"))  # noqa: S614

        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(self.hyperparameters, f)

        joblib.dump(self.gene_expression_scaler, os.path.join(directory, "gene_scaler.pkl"))
        joblib.dump(self.methylation_scaler, os.path.join(directory, "methylation_scaler.pkl"))
        joblib.dump(self.methylation_pca, os.path.join(directory, "methylation_pca.pkl"))

        metadata = {
            "dim_gex": self.dim_gex,
            "dim_met": self.dim_met,
            "dim_mut": self.dim_mut,
            "dim_cnv": self.dim_cnv,
            "dim_fp": self.dim_fp,
        }
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, directory: str) -> "MultiOmicsNeuralNetwork":
        """
        Load a trained MultiOmicsNeuralNetwork instance from disk.

        Required files:
        - model.pt
        - hyperparameters.json
        - gene_scaler.pkl
        - methylation_scaler.pkl
        - methylation_pca.pkl
        - metadata.json

        :param directory: Directory containing the saved model files
        :return: Fully restored MultiOmicsNeuralNetwork instance
        :raises FileNotFoundError: if any required file is missing
        """
        required_files = [
            "model.pt",
            "hyperparameters.json",
            "gene_scaler.pkl",
            "methylation_scaler.pkl",
            "methylation_pca.pkl",
            "metadata.json",
        ]
        missing = [f for f in required_files if not os.path.exists(os.path.join(directory, f))]
        if missing:
            raise FileNotFoundError(f"Missing model files: {', '.join(missing)}")

        instance = cls()

        with open(os.path.join(directory, "hyperparameters.json")) as f:
            instance.hyperparameters = json.load(f)

        instance.gene_expression_scaler = joblib.load(os.path.join(directory, "gene_scaler.pkl"))
        instance.methylation_scaler = joblib.load(os.path.join(directory, "methylation_scaler.pkl"))
        instance.methylation_pca = joblib.load(os.path.join(directory, "methylation_pca.pkl"))

        with open(os.path.join(directory, "metadata.json")) as f:
            metadata = json.load(f)

        instance.dim_gex = metadata["dim_gex"]
        instance.dim_met = metadata["dim_met"]
        instance.dim_mut = metadata["dim_mut"]
        instance.dim_cnv = metadata["dim_cnv"]
        instance.dim_fp = metadata["dim_fp"]

        input_dim = instance.dim_gex + instance.dim_met + instance.dim_mut + instance.dim_cnv + instance.dim_fp

        instance.model = FeedForwardNetwork(
            hyperparameters=instance.hyperparameters,
            input_dim=input_dim,
        )
        instance.model.load_state_dict(torch.load(os.path.join(directory, "model.pt")))  # noqa: S614
        instance.model.eval()

        return instance
