import numpy as np
from sklearn.svm import SVR
from drevalpy.datasets.dataset import FeatureDataset, DrugResponseDataset
from drevalpy.models.drp_model import DRPModel
from ..utils import (
    load_ge_features_from_landmark_genes,
    load_drug_features_from_fingerprints,
)


class SVMRegressor(DRPModel):
    model_name = "SVR"
    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.
        :param hyperparameters: Hyperparameters for the model.
        """
        self.model = SVR(
            kernel=hyperparameters["kernel"],
            C=hyperparameters["C"],
            epsilon=hyperparameters["epsilon"],
            max_iter=hyperparameters["max_iter"],
        )

    def train(
        self,
        output: DrugResponseDataset,
        gene_expression: np.ndarray = None,
        fingerprints: np.ndarray = None,
    ) -> None:
        """
        Trains the model: the number of features is the number of genes + the number of fingerprints.
        :param output: training dataset containing the response output
        :param gene_expression: training dataset containing gene expression data
        :param fingerprints: training dataset containing fingerprints data
        """
        X = np.concatenate((gene_expression, fingerprints), axis=1)
        self.model.fit(X, output.response)

    def predict(
        self, gene_expression: np.ndarray = None, fingerprints: np.ndarray = None
    ) -> np.ndarray:
        """
        Predicts the response values.
        :param gene_expression:
        :param fingerprints:
        :return:
        """
        X = np.concatenate((gene_expression, fingerprints), axis=1)
        return self.model.predict(X)

    def save(self, path):
        raise NotImplementedError("SVR does not support saving yet ...")

    def load(self, path):
        raise NotImplementedError("SVR does not support loading yet ...")

    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
        """
        Loads the cell line features.
        :param path: Path to the gene expression and landmark genes
        :return: FeatureDataset containing the cell line gene expression features, filtered through the landmark genes
        """
        return load_ge_features_from_landmark_genes(data_path, dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        return load_drug_features_from_fingerprints(data_path, dataset_name)
