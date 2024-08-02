from typing import Optional
from drevalpy.models.utils import (
    load_drug_features_from_fingerprints,
    load_and_reduce_gene_features,
)
from drevalpy.models.SimpleNeuralNetwork.utils import FeedForwardNetwork
from drevalpy.models.drp_model import DRPModel
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
import numpy as np
import warnings
from numpy.typing import ArrayLike

class SimpleNeuralNetwork(DRPModel):
    """
    Simple Feedforward Neural Network model with dropout.
    hyperparameters:
        units_per_layer: number of units per layer e.g. [100, 50] means 2 layers with 100 and 50 units respectively and the output layer with one unit.
        dropout_prob: dropout probability for layers 1, 2, ..., n-1
    """

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]
    early_stopping = True
    model_name = "SimpleNeuralNetwork"

    def build_model(self, hyperparameters: dict):
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
        output_earlystopping: Optional[DrugResponseDataset] = None,
        cell_line_input: FeatureDataset = None,
        drug_input: FeatureDataset = None,
    ):
        """
        Trains the model.
        :param output: training data associated with the response output
        :param output_earlystopping: optional early stopping dataset
        :param gene_expression: gene expression data
        :param fingerprints: fingerprints data
        :param gene_expression_earlystopping: gene expression data for early stopping
        :param fingerprints_earlystopping: fingerprints data for early stopping

        """
        X = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input
        )

        if output_earlystopping is not None:
            X_earlystopping = self.get_concatenated_features(
                cell_line_view="gene_expression",
                drug_view="fingerprints",
                cell_line_ids_output=output_earlystopping.cell_line_ids,
                drug_ids_output=output_earlystopping.drug_ids,
                cell_line_input=cell_line_input,
                drug_input=drug_input
            )

        else:
            X_earlystopping = None

        if output_earlystopping is not None:
            response_earlystopping = output_earlystopping.response
        else:
            response_earlystopping = None

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*does not have many workers which may be a bottleneck.*",
            )
            self.model.fit(
                X_train=X,
                y_train=output.response,
                X_eval=X_earlystopping,
                y_eval=response_earlystopping,
                batch_size=16,
                patience=5,
                num_workers=1,
            )

    def save(self, path: str):
        """
        Saves the model.
        :param path: path to save the model
        """
        self.model.save(path)

    @staticmethod
    def load(path: str):
        # TODO
        raise NotImplementedError("load method not implemented")

    def predict(
        self, cell_line_ids: ArrayLike, drug_ids: ArrayLike, cell_line_input: FeatureDataset, drug_input: FeatureDataset
    ) -> np.ndarray:
        """
        Predicts the response for the given input.
        """
        X = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input)
        
        return self.model.predict(X)

    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
        """
        Loads the cell line features.
        :param path: Path to the gene expression and landmark genes
        :return: FeatureDataset containing the cell line gene expression features, filtered through the landmark genes
        """

        return load_and_reduce_gene_features(
            feature_type="gene_expression",
            gene_list="landmark_genes",
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:

        return load_drug_features_from_fingerprints(data_path, dataset_name)
