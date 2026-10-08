"""Contains the Multi-OMICS Random Forest model."""

import os

import joblib
import numpy as np
from sklearn.decomposition import PCA

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..utils import _get_view_as_list, load_multi_cell_line_view
from .sklearn_models import RandomForest


class MultiViewRandomForest(RandomForest):
    """Multi-View Random Forest model."""

    cell_line_views = [
        "gene_expression",
        "methylation",
        "mutations",
        "copy_number_variation_gistic",
    ]
    #: Per-omic override of the default gene list, e.g. {"gene_expression": "landmark_genes"}.
    #: Overridable via the "gene_lists" hyperparameter.
    gene_lists: dict[str, str | None] | None = None

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: MultiViewRandomForest
        """
        return "MultiViewRandomForest"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        :param hyperparameters: Hyperparameters for the model, see RandomForest.build_model. Additionally supports
            "gene_lists", a per-omic override of the default gene list used to subset cell line features, and
            "view_configs", a list of {"cell_line_views": [...], "gene_lists": {...}} entries to grid-search
            cell_line_views and gene_lists together instead of independently. When present, "view_configs"
            takes precedence over "cell_line_views" and "gene_lists".
        """
        super().build_model(hyperparameters)
        view_config = hyperparameters.get("view_configs")
        if view_config is not None:
            self.cell_line_views = _get_view_as_list(view_config["cell_line_views"])
            self.gene_lists = view_config.get("gene_lists")
        else:
            self.gene_lists = hyperparameters.get("gene_lists", type(self).gene_lists)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features for a multi-view random forest.

        :param data_path: data path e.g. data/
        :param dataset_name: dataset name e.g. GDSC1
        :returns: FeatureDataset containing the cell line omics features
        """
        return load_multi_cell_line_view(
            self.cell_line_views, data_path, dataset_name, self.get_model_name(), gene_lists=self.gene_lists
        )

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Trains the model: the number of features is the number of genes + the number of fingerprints.

        :param output: training dataset containing the response output
        :param cell_line_input: training dataset containing the OMICs
        :param drug_input: training dataset containing fingerprints data
        :param output_earlystopping: not needed
        :param model_checkpoint_dir: not needed
        """
        inputs = self.get_feature_matrices(
            cell_line_ids=output.cell_line_ids,
            drug_ids=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        # concatenate in the order of self.cell_line_views
        array_list = []
        for view in self.cell_line_views:
            feature_mat = inputs[view]

            if view == "methylation":
                # feature_mat is response-level, so fit on each cell line once rather than once per drug
                unique_cell_line_features = cell_line_input.get_feature_matrix(
                    view=view, identifiers=np.unique(output.cell_line_ids)
                )
                n_components = min(self.methylation_n_components, *unique_cell_line_features.shape)
                self.methylation_pca = PCA(n_components=n_components)
                self.methylation_pca.fit(unique_cell_line_features)
                feature_mat = self.methylation_pca.transform(feature_mat)

            array_list.append(feature_mat)

        x = np.concatenate(array_list, axis=1)
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
        :raises RuntimeError: if PCA has not been fit
        """
        if "methylation" in self.cell_line_views and not hasattr(self.methylation_pca, "components_"):
            raise RuntimeError("PCA has not been fit. Call train() before predict().")

        inputs = self.get_feature_matrices(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        # concatenate in the order of self.cell_line_views
        array_list = []
        for view in self.cell_line_views:
            feature_mat = inputs[view]

            if view == "methylation":
                feature_mat = self.methylation_pca.transform(feature_mat)

            array_list.append(feature_mat)

        x = np.concatenate(array_list, axis=1)

        return self.model.predict(x)

    def save(self, directory: str) -> None:
        """
        Saves the trained model, hyperparameters, scaler, and PCA transformer to the specified directory.

        :param directory: Path to the directory where model components will be saved.
        """
        super().save(directory)
        if self.methylation_pca is not None:
            joblib.dump(self.methylation_pca, os.path.join(directory, "pca.pkl"))

    @classmethod
    def load(cls, directory: str) -> "MultiViewRandomForest":
        """
        Loads the trained model, hyperparameters, scaler, and PCA transformer from the specified directory.

        :param directory: Path to the directory where model components are stored.
        :returns: An instance of MultiViewRandomForest with restored state.
        """
        instance: MultiViewRandomForest = super().load(directory)  # type: ignore[assignment]
        pca_path = os.path.join(directory, "pca.pkl")
        if os.path.exists(pca_path):
            instance.methylation_pca = joblib.load(pca_path)
        return instance
