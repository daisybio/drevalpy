"""
Contains sklearn baseline models: ElasticNet, RandomForest, SVM
"""

from typing import Dict
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR
from numpy.typing import ArrayLike

from drevalpy.datasets.dataset import FeatureDataset, DrugResponseDataset
from drevalpy.models.drp_model import DRPModel
from ..utils import (
    load_and_reduce_gene_features,
    load_drug_fingerprint_features,
)


class SklearnModel(DRPModel):
    """
    Parent class that contains the common methods for the sklearn models.
    """

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]

    def __init__(self):
        super().__init__()
        self.model = None

    def build_model(self, hyperparameters: Dict):
        """
        Builds the model from hyperparameters.
        :param hyperparameters: Custom hyperparameters for the model, have to be defined in the
        child class.
        """
        raise NotImplementedError(
            "build_model method has to be implemented in the child class."
        )

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset = None,
        output_earlystopping=None,
    ) -> None:
        """
        Trains the model: the number of features is the number of genes + the number of
        fingerprints.
        :param output: training dataset containing the response output
        :param cell_line_input: training dataset containing gene expression data
        :param drug_input: training dataset containing fingerprints data
        :param output_earlystopping: not needed
        """

        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        self.model.fit(x, output.response)

    def predict(
        self,
        drug_ids: ArrayLike,
        cell_line_ids: ArrayLike,
        drug_input: FeatureDataset = None,
        cell_line_input: FeatureDataset = None,
    ) -> np.ndarray:
        """
        Predicts the response for the given input.
        :param gene_expression: gene expression data
        :param fingerprints: fingerprints data
        :return: predicted response
        """
        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        return self.model.predict(x)

    def save(self, path):
        raise NotImplementedError("ElasticNetModel does not support saving yet ...")

    def load(self, path):
        raise NotImplementedError("ElasticNetModel does not support loading yet ...")

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


class ElasticNetModel(SklearnModel):
    """
    ElasticNet model for drug response prediction.
    """

    model_name = "ElasticNet"

    def build_model(self, hyperparameters: Dict):
        """
        Builds the ElasticNet model from hyperparameters.
        :param hyperparameters: Contains L1 ratio and alpha.
        """
        if hyperparameters["l1_ratio"] == 0.0:
            self.model = Ridge(alpha=hyperparameters["alpha"])
        else:
            self.model = ElasticNet(
                alpha=hyperparameters["alpha"], l1_ratio=hyperparameters["l1_ratio"]
            )


class RandomForest(SklearnModel):
    """
    RandomForest model for drug response prediction.
    """

    model_name = "RandomForest"

    def build_model(self, hyperparameters: Dict):
        """
        Builds the model from hyperparameters.
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


class SVMRegressor(SklearnModel):
    """
    SVM model for drug response prediction.
    """

    model_name = "SVR"

    def build_model(self, hyperparameters: Dict):
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


class GradientBoosting(SklearnModel):
    """
    Gradient Boosting model for drug response prediction.
    """

    model_name = "GradientBoosting"

    def build_model(self, hyperparameters: Dict):
        """
        Builds the model from hyperparameters.
        :param hyperparameters: Hyperparameters for the model.
        """
        self.model = GradientBoostingRegressor(
            n_estimators=hyperparameters.get("n_estimators", 100),
            learning_rate=hyperparameters.get("learning_rate", 0.1),
            max_depth=hyperparameters.get("max_depth", 3),
            min_samples_split=hyperparameters.get("min_samples_split", 2),
            min_samples_leaf=hyperparameters.get("min_samples_leaf", 1),
            subsample=hyperparameters.get("subsample", 1.0),
            max_features=hyperparameters.get("max_features", 1.0),
        )
