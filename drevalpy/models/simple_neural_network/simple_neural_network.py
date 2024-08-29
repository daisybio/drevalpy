"""
Contains the SimpleNeuralNetwork model.
"""
import warnings
from typing import Optional, Dict
import numpy as np
from numpy.typing import ArrayLike

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from ..utils import (
    load_drug_features_from_fingerprints,
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
            cell_line_input = self._fit_transform_gene_expression_features(cell_line_input, output.cell_line_ids)
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
        # Apply the transformation to gene expression features before prediction
        if "gene_expression" in self.cell_line_views:
            cell_line_input = self._transform_gene_expression_features(cell_line_input=cell_line_input, cell_line_ids=cell_line_ids)

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

        return load_drug_features_from_fingerprints(data_path, dataset_name)
    
    def _transform_gene_expression_features(self, cell_line_input: FeatureDataset, cell_line_ids: ArrayLike) -> FeatureDataset:
        """
        Applies arcsinh transformation and scaling to gene expression features during prediction.
        """
        cell_line_input = cell_line_input.copy()
        for cell_line_id in cell_line_ids:
            gene_expression = cell_line_input.features[cell_line_id]["gene_expression"]
            transformed_gene_expression = np.arcsinh(gene_expression)
            scaled_gene_expression = self.gene_expression_scaler.transform([transformed_gene_expression])[0]
            cell_line_input.features[cell_line_id]["gene_expression"] = scaled_gene_expression

        return cell_line_input
    
    def _fit_transform_gene_expression_features(self, cell_line_input: FeatureDataset, train_cell_line_ids) -> FeatureDataset:
        """
        Applies arcsinh transformation and scaling to gene expression features for each cell line. Only train cell lines are used for fiiting the Scaler.
        :param cell_line_input: The feature dataset containing cell line features.
        :param cell_line_ids: The cell line IDs corresponding to the training dataset.
        :return: The modified FeatureDataset with transformed gene expression features.
        """
        cell_line_input = cell_line_input.copy()

        train_gene_expression = []

        # Collect all gene expression features for fitting the scaler
        for cell_line_id in train_cell_line_ids:
            gene_expression = cell_line_input.features[cell_line_id]["gene_expression"]
            transformed_gene_expression = np.arcsinh(gene_expression)
            train_gene_expression.append(transformed_gene_expression)

        # Fit the scaler on the collected gene expression data
        train_gene_expression = np.vstack(train_gene_expression)
        self.gene_expression_scaler.fit(train_gene_expression)

        # Apply transformation and scaling to each cell line's gene expression data
        for cell_line_id in cell_line_input.features:
            gene_expression = cell_line_input.features[cell_line_id]["gene_expression"]
            transformed_gene_expression = np.arcsinh(gene_expression)
            scaled_gene_expression = self.gene_expression_scaler.transform([transformed_gene_expression])[0]
            cell_line_input.features[cell_line_id]["gene_expression"] = scaled_gene_expression

        return cell_line_input

