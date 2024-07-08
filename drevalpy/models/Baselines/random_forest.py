import numpy as np
from sklearn.ensemble import RandomForestRegressor

from drevalpy.datasets.dataset import FeatureDataset, DrugResponseDataset
from drevalpy.models.drp_model import DRPModel
from ..utils import (
    load_and_reduce_gene_features,
    load_drug_features_from_fingerprints,
)


class RandomForest(DRPModel):
    model_name = "RandomForest"
    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]

    def build_model(self, hyperparameters: dict, **kwargs):
        """
        Builds the model from hyperparameters.
        :param **kwargs:
        :param hyperparameters: Hyperparameters for the model.
        """
        if hyperparameters["max_depth"] == "None":
            hyperparameters["max_depth"] = None
        self.model = RandomForestRegressor(
            n_estimators=hyperparameters["n_estimators"],
            criterion=hyperparameters["criterion"],
            max_depth=hyperparameters["max_depth"],
            min_samples_split=hyperparameters["min_samples_split"],
            min_samples_leaf=hyperparameters["min_samples_leaf"],
            n_jobs=hyperparameters["n_jobs"],
            max_samples=hyperparameters["max_samples"],
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
        Predicts the response for the given input.
        :param gene_expression: gene expression data
        :param fingerprints: fingerprints data
        :return: predicted response
        """
        X = np.concatenate((gene_expression, fingerprints), axis=1)
        return self.model.predict(X)

    def save(self, path):
        raise NotImplementedError("RF does not support saving yet ...")

    def load(self, path):
        raise NotImplementedError("RF does not support loading yet ...")

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
