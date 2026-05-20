"""Contains the baseline MultiViewNeuralNetwork model."""

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
from ..utils import (
    _get_view_as_list,
    load_multi_cell_line_view,
    load_single_drug_view,
    prepare_expression_and_methylation,
)
from .utils import FeedForwardNetwork


class MultiViewNeuralNetwork(DRPModel):
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
        Initalization method for MultiViewNeuralNetwork Model.

        The PCA is initialized to None because it depends on hyperparameter, therefore built in build_model.
        """
        super().__init__()
        self.model = None
        self.methylation_scaler = StandardScaler()
        self.methylation_pca = None
        self.pca_ncomp = 100
        self.gene_expression_scaler = StandardScaler()
        self.input_dims = dict()

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: MultiViewNeuralNetwork
        """
        return "MultiViewNeuralNetwork"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        The model is a simple feedforward neural network with dropout. The PCA is used to reduce the dimensionality of
        the methylation data.

        :param hyperparameters: dictionary containing the hyperparameters units_per_layer, dropout_prob, and
            methylation_pca_components.
        """
        # Log hyperparameters to wandb if enabled
        self.log_hyperparameters(hyperparameters)

        self.hyperparameters = hyperparameters
        self.cell_line_views = _get_view_as_list(
            hyperparameters.get(
                "cell_line_views", ["gene_expression", "methylation", "mutations", "copy_number_variation_gistic"]
            )
        )
        self.drug_views = _get_view_as_list(hyperparameters.get("drug_views", ["fingerprints"]))
        if "methylation" in self.cell_line_views:
            self.pca_ncomp = hyperparameters["methylation_pca_components"]

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features for a multi-view neural network.

        :param data_path: data path e.g. data/
        :param dataset_name: dataset name e.g. GDSC1
        :returns: FeatureDataset containing the cell line omics features
        """
        return load_multi_cell_line_view(self.cell_line_views, data_path, dataset_name, self.get_model_name())

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        Load the drug features for a multi-view neural network.

        :param data_path: path to the drug features, e.g., data/
        :param dataset_name: name of the dataset, e.g., GDSC1
        :returns: FeatureDataset containing the drug features
        """
        return load_single_drug_view(self.drug_views, data_path, dataset_name, self.get_model_name())

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
        :raises ValueError: if drug_input is missing
        """
        if drug_input is None:
            raise ValueError(f"Drug input ({self.drug_views[0]}) is needed for the MultiViewNeuralNetwork model.")
        first_cl_feature = next(iter(cell_line_input.features.values()))
        if "methylation" in self.cell_line_views:
            n_met_features = first_cl_feature["methylation"].shape[0]
            if n_met_features > self.pca_ncomp:
                self.methylation_pca = PCA(n_components=self.pca_ncomp)
            else:
                self.methylation_pca = PCA(n_components=n_met_features)

        # if gene expression or methylation don't even occur, this just returns cell_line_input, so it's fine
        cell_line_input = prepare_expression_and_methylation(
            cell_line_input=cell_line_input,
            cell_line_ids=np.unique(output.cell_line_ids),
            training=True,
            gene_expression_scaler=self.gene_expression_scaler,
            methylation_scaler=self.methylation_scaler,
            methylation_pca=self.methylation_pca,
        )
        first_drug_feature = next(iter(drug_input.features.values()))

        cell_line_dims = {
            view: first_cl_feature[view].shape[0] for view in self.cell_line_views if view != "methylation"
        }
        if "methylation" in self.cell_line_views:
            cell_line_dims["methylation"] = self.methylation_pca.n_components

        drug_dims = {view: first_drug_feature[view].shape[0] for view in self.drug_views}

        self.input_dims = {**cell_line_dims, **drug_dims}
        total_dim = sum(self.input_dims.values())

        self.model = FeedForwardNetwork(
            hyperparameters=self.hyperparameters,
            input_dim=total_dim,
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
                wandb_project=self.wandb_project,
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

        # concatenate in the order of self.cell_line_views
        array_list = []
        for view in self.cell_line_views + self.drug_views:
            feature_mat = inputs[view]
            array_list.append(feature_mat)

        x = np.concatenate(array_list, axis=1)
        return self.model.predict(x)

    def save(self, directory: str) -> None:
        """
        Save the trained model, hyperparameters, scalers, PCA object, and feature dimensions to disk.

        Files always saved: model.pt, hyperparameters.json, metadata.json.
        Conditionally saved: gene_scaler.pkl (if gene_expression in views),
        methylation_scaler.pkl and methylation_pca.pkl (if methylation in views).

        :param directory: Target directory
        """
        os.makedirs(directory, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(directory, "model.pt"))  # noqa: S614

        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(self.hyperparameters, f)

        if "gene_expression" in self.cell_line_views:
            joblib.dump(self.gene_expression_scaler, os.path.join(directory, "gene_scaler.pkl"))
        if "methylation" in self.cell_line_views:
            joblib.dump(self.methylation_scaler, os.path.join(directory, "methylation_scaler.pkl"))
            joblib.dump(self.methylation_pca, os.path.join(directory, "methylation_pca.pkl"))

        metadata = {
            "input_dims": self.input_dims,
        }
        with open(os.path.join(directory, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, directory: str) -> "MultiViewNeuralNetwork":
        """
        Load a trained MultiViewNeuralNetwork instance from disk.

        Always required: model.pt, hyperparameters.json, metadata.json.
        Conditionally required: gene_scaler.pkl (if gene_expression in views),
        methylation_scaler.pkl and methylation_pca.pkl (if methylation in views).

        :param directory: Directory containing the saved model files
        :return: Fully restored MultiViewNeuralNetwork instance
        :raises FileNotFoundError: if any required file is missing
        """
        instance = cls()

        with open(os.path.join(directory, "hyperparameters.json")) as f:
            hyperparameters = json.load(f)

        instance.build_model(hyperparameters)

        required_files = ["model.pt", "hyperparameters.json", "metadata.json"]
        if "gene_expression" in instance.cell_line_views:
            required_files.append("gene_scaler.pkl")
        if "methylation" in instance.cell_line_views:
            required_files.extend(["methylation_scaler.pkl", "methylation_pca.pkl"])

        missing = [f for f in required_files if not os.path.exists(os.path.join(directory, f))]
        if missing:
            raise FileNotFoundError(f"Missing model files: {', '.join(missing)}")

        if "gene_expression" in instance.cell_line_views:
            instance.gene_expression_scaler = joblib.load(os.path.join(directory, "gene_scaler.pkl"))
        if "methylation" in instance.cell_line_views:
            instance.methylation_scaler = joblib.load(os.path.join(directory, "methylation_scaler.pkl"))
            instance.methylation_pca = joblib.load(os.path.join(directory, "methylation_pca.pkl"))

        with open(os.path.join(directory, "metadata.json")) as f:
            metadata = json.load(f)

        instance.input_dims = metadata["input_dims"]
        total_dim = sum(instance.input_dims.values())

        instance.model = FeedForwardNetwork(
            hyperparameters=instance.hyperparameters,
            input_dim=total_dim,
        )
        instance.model.load_state_dict(torch.load(os.path.join(directory, "model.pt")))  # noqa: S614
        instance.model.eval()

        return instance
