"""
Contains the MultiOmicsNeuralNetwork model.
"""

import warnings
from typing import Optional, Dict
import numpy as np
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from .utils import FeedForwardNetwork
from ..drp_model import DRPModel
from ..utils import (
    load_drug_fingerprint_features,
    get_multiomics_feature_dataset,
)


class MultiOmicsNeuralNetwork(DRPModel):
    """
    Simple Feedforward Neural Network model with dropout.
    hyperparameters:
        units_per_layer: number of units per layer e.g. [100, 50] means 2 layers with 100 and 50
        units respectively and the output layer with one unit.
        dropout_prob: dropout probability for layers 1, 2, ..., n-1
    """

    cell_line_views = [
        "gene_expression",
        "methylation",
        "mutations",
        "copy_number_variation_gistic",
    ]
    drug_views = ["fingerprints"]
    early_stopping = True
    model_name = "MultiOmicsNeuralNetwork"

    def __init__(self):
        super().__init__()
        self.model = None
        self.pca = None

    def build_model(self, hyperparameters: Dict):
        """
        Builds the model from hyperparameters.
        """
        self.model = FeedForwardNetwork(
            n_units_per_layer=hyperparameters["units_per_layer"],
            dropout_prob=hyperparameters["dropout_prob"],
        )
        self.pca = PCA(n_components=hyperparameters["methylation_pca_components"])

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
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
        unique_methylation = np.stack(
            [
                cell_line_input.features[id_]["methylation"]
                for id_ in np.unique(output.cell_line_ids)
            ],
            axis=0,
        )

        self.pca.n_components = min(self.pca.n_components, len(unique_methylation))
        self.pca = self.pca.fit(unique_methylation)

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
                met_transform=self.pca,
            )

    def save(self, path: str):
        """
        Saves the model.
        :param path: path to save the model
        """
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
        inputs = self.get_feature_matrices(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        (
            gene_expression,
            methylation,
            mutations,
            copy_number_variation_gistic,
            fingerprints,
        ) = (
            inputs["gene_expression"],
            inputs["methylation"],
            inputs["mutations"],
            inputs["copy_number_variation_gistic"],
            inputs["fingerprints"],
        )
        methylation = self.pca.transform(methylation)
        x = np.concatenate(
            (
                gene_expression,
                methylation,
                mutations,
                copy_number_variation_gistic,
                fingerprints,
            ),
            axis=1,
        )
        return self.model.predict(x)

    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
        """
        Loads the cell line features.
        :param data_path: data path e.g. data/
        :param dataset_name: dataset name e.g. GDSC1

        :return: FeatureDataset containing the cell line omics features, filtered through the
        drug target genes
        """

        return get_multiomics_feature_dataset(
            data_path=data_path, dataset_name=dataset_name
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        return load_drug_fingerprint_features(data_path, dataset_name)
