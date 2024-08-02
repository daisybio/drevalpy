import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from numpy.typing import ArrayLike

from drevalpy.datasets.dataset import FeatureDataset, DrugResponseDataset
from drevalpy.models.drp_model import DRPModel
from ..utils import (
    load_and_reduce_gene_features,
    load_drug_features_from_fingerprints,
)


class ElasticNetModel(DRPModel):
    model_name = "ElasticNet"
    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]

    def build_model(self, hyperparameters: dict, *args, **kwargs):
        """
        Builds the model from hyperparameters.
        :param hyperparameters: Hyperparameters for the model.
        """
        if hyperparameters["l1_ratio"] == 0.0:
            self.model = Ridge(alpha=hyperparameters["alpha"])
        else:
            self.model = ElasticNet(
                alpha=hyperparameters["alpha"], l1_ratio=hyperparameters["l1_ratio"]
            )

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset = None,
        *args,
        **kwargs
    ) -> None:
        """
        Trains the model: the number of features is the number of genes + the number of fingerprints.
        :param output: training dataset containing the response output
        :param cell_line_input: training dataset containing gene expression data
        :param drug_input: training dataset containing fingerprints data
        """
        X = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        self.model.fit(X, output.response)

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
        X = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view="fingerprints",
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        return self.model.predict(X)

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
