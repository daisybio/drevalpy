"""
Contains the SimpleNeuralNetwork model.
"""

import warnings
from typing import Optional, Dict
import numpy as np
from numpy.typing import ArrayLike

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from ..utils import (
    load_drug_fingerprint_features,
    load_and_reduce_gene_features,
)
from .utils import FeedForwardNetwork
from ..drp_model import DRPModel
from sklearn.preprocessing import StandardScaler


class SimpleNeuralNetwork(DRPModel):
    """
    Simple Feedforward Neural Network model with dropout.
    hyperparameters:
        units_per_layer: number of units per layer e.g. [100, 50] means 2 layers with 100 and 50
        units respectively and the output layer with one unit.
        dropout_prob: dropout probability for layers 1, 2, ..., n-1
    """

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]
    early_stopping = True
    model_name = "SimpleNeuralNetwork"

    def __init__(self):
        super().__init__()
        self.model = None
        self.gene_expression_scaler = StandardScaler()

    def build_model(self, hyperparameters: Dict):
        """
        Builds the model from hyperparameters.
        """
        self.model = FeedForwardNetwork(
            n_units_per_layer=hyperparameters["units_per_layer"],
            dropout_prob=hyperparameters["dropout_prob"],
        )

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset = None,
        drug_input: FeatureDataset = None,
        output_earlystopping: Optional[DrugResponseDataset] = None,
    ):
        """
        Trains the model.
        :param output: training data associated with the response output
        :param cell_line_input: cell line omics features
        :param drug_input: drug omics features
        :param output_earlystopping: optional early stopping dataset

        """
        # Apply arcsinh transformation and scaling to gene expression features
        if "gene_expression" in self.cell_line_views:
            cell_line_input = cell_line_input.copy()
            cell_line_input.apply(function=np.arcsinh, view="gene_expression")
            self.gene_expression_scaler = cell_line_input.fit_transform_features(
                train_ids=np.unique(output.cell_line_ids),
                transformer=self.gene_expression_scaler,
                view="gene_expression",
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
                batch_size=16,
                patience=5,
                num_workers=1,
            )

    def save(self, path: str):
        raise NotImplementedError("save method not implemented")

    def load(self, path: str):
        raise NotImplementedError("load method not implemented")

    def predict(
        self,
        drug_ids: ArrayLike,
        cell_line_ids: ArrayLike,
        drug_input: FeatureDataset = None,
        cell_line_input: FeatureDataset = None,
    ) -> np.ndarray:
        """
        Predicts the response for the given input.
        """
        # Apply transformation to gene expression features before prediction
        if "gene_expression" in self.cell_line_views:
            cell_line_input = cell_line_input.copy()
            cell_line_input.apply(function=np.arcsinh, view="gene_expression")
            cell_line_input.transform_features(
                ids=np.unique(cell_line_ids),
                transformer=self.gene_expression_scaler,
                view="gene_expression",
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

    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
        """
        Loads the cell line features.
        :param path: Path to the gene expression and landmark genes
        :return: FeatureDataset containing the cell line gene expression features, filtered
        through the landmark genes
        """

        return load_and_reduce_gene_features(
            feature_type="gene_expression",
            gene_list="landmark_genes",
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:

        return load_drug_fingerprint_features(data_path, dataset_name)
