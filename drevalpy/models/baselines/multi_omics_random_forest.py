"""Contains the Multi-OMICS Random Forest model."""

import numpy as np
from sklearn.decomposition import PCA

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..utils import get_multiomics_feature_dataset
from .sklearn_models import RandomForest


class MultiOmicsRandomForest(RandomForest):
    """Multi-OMICS Random Forest model."""

    cell_line_views = [
        "gene_expression",
        "methylation",
        "mutations",
        "copy_number_variation_gistic",
    ]

    def __init__(self):
        """
        Initializes the model.

        Sets the PCA to None, which is initialized in the build_model method.
        """
        super().__init__()
        self.pca = None

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: MultiOmicsRandomForest
        """
        return "MultiOmicsRandomForest"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        :param hyperparameters: Hyperparameters for the model.
        """
        super().build_model(hyperparameters)
        self.pca = PCA(n_components=hyperparameters["n_components"])

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: data path e.g. data/
        :param dataset_name: dataset name e.g. GDSC1
        :returns: FeatureDataset containing the cell line omics features, filtered through the
            drug target genes
        """
        return get_multiomics_feature_dataset(data_path=data_path, dataset_name=dataset_name)

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
    ) -> None:
        """
        Trains the model: the number of features is the number of genes + the number of fingerprints.

        :param output: training dataset containing the response output
        :param cell_line_input: training dataset containing the OMICs
        :param drug_input: training dataset containing fingerprints data
        :param output_earlystopping: not needed
        """
        inputs = self.get_feature_matrices(
            cell_line_ids=output.cell_line_ids,
            drug_ids=output.drug_ids,
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
        methylation = self.pca.fit_transform(methylation)

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
        self.model.fit(x, output.response)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the response for the given input.

        :param cell_line_ids: cell line ids
        :param drug_ids: drug ids
        :param cell_line_input: cell line input
        :param drug_input: drug input
        :returns: predicted response
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
