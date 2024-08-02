import numpy as np
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

from drevalpy.datasets.dataset import FeatureDataset, DrugResponseDataset
from drevalpy.models.drp_model import DRPModel
from ..utils import (
    get_multiomics_feature_dataset,
    load_and_reduce_gene_features,
    load_drug_features_from_fingerprints,
)


class RandomForest(DRPModel):
    model_name = "RandomForest"
    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]

    def build_model(self, hyperparameters: dict, *args, **kwargs):
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
        :param drug_ids: drug ids
        :param cell_line_ids: cell line ids
        :param drug_input: drug input
        :param cell_line_input: cell line input
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


class MultiOmicsRandomForest(RandomForest):
    cell_line_views = [
        "gene_expression",
        "methylation",
        "mutations",
        "copy_number_variation_gistic",
    ]
    model_name = "MultiOmicsRandomForest"

    def build_model(self, hyperparameters: dict, *args, **kwargs):
        """
        Builds the model from hyperparameters.
        :param hyperparameters: Hyperparameters for the model.
        """
        super().build_model(hyperparameters)
        self.pca = PCA(n_components=hyperparameters["n_components"])

    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
        """
        Loads the cell line features.
        :param data_path: data path e.g. data/
        :param dataset_name: dataset name e.g. GDSC1

        :return: FeatureDataset containing the cell line omics features, filtered through the drug target genes
        """

        return get_multiomics_feature_dataset(
            data_path=data_path, dataset_name=dataset_name
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
        :param cell_line_input: training dataset containing the OMICs
        :param drug_input: training dataset containing fingerprints data
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

        X = np.concatenate(
            (
                gene_expression,
                methylation,
                mutations,
                copy_number_variation_gistic,
                fingerprints,
            ),
            axis=1,
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
        X = np.concatenate(
            (
                gene_expression,
                methylation,
                mutations,
                copy_number_variation_gistic,
                fingerprints,
            ),
            axis=1,
        )
        return self.model.predict(X)
