"""Contains the baseline MultiViewLightGBM model."""

import json
import os

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset

from ..drp_model import DRPModel
from ..utils import (
    ProteomicsMedianCenterAndImputeTransformer,
    _get_view_as_list,
    load_multi_cell_line_view,
    load_single_drug_view,
    prepare_expression_and_methylation,
    prepare_proteomics,
)


class MultiViewLightGBM(DRPModel):
    """LightGBM model with multi-omic cell line features and drug fingerprints."""

    cell_line_views = [
        "gene_expression",
        "methylation",
        "mutations",
        "copy_number_variation_gistic",
    ]
    drug_views = ["fingerprints"]

    def __init__(self):
        """Initializes the MultiViewLightGBM model."""
        super().__init__()
        self.model = None
        self.gene_expression_scaler = StandardScaler()
        # methylation-specific defaults
        self.methylation_scaler = StandardScaler()
        self.methylation_pca = None
        self.pca_ncomp = 100
        # proteomics-specific defaults
        self.proteomics_transformer = None
        self.proteomics_feature_threshold = 0.7
        self.proteomics_n_features = 1000
        self.proteomics_normalization_width = 0.3
        self.proteomics_normalization_downshift = 1.8

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: MultiViewLightGBM
        """
        return "MultiViewLightGBM"

    def build_model(self, hyperparameters: dict) -> None:
        """
        Builds the model from hyperparameters.

        :param hyperparameters: dictionary containing the hyperparameters.
        :raises ImportError: if lightgbm is not installed.
        """
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "MultiViewLightGBM requires the optional 'lightgbm' extra. "
                "Install it with: pip install drevalpy[lightgbm] (or `poetry install -E lightgbm`)."
            ) from e

        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = hyperparameters
        self.cell_line_views = _get_view_as_list(
            hyperparameters.get(
                "cell_line_views",
                ["gene_expression", "methylation", "mutations", "copy_number_variation_gistic"],
            )
        )
        self.drug_views = _get_view_as_list(hyperparameters.get("drug_views", ["fingerprints"]))
        if "methylation" in self.cell_line_views:
            self.pca_ncomp = hyperparameters.get("methylation_n_components", 100)
        if "proteomics" in self.cell_line_views:
            self.proteomics_feature_threshold = hyperparameters.get("proteomics_feature_threshold", 0.7)
            self.proteomics_n_features = hyperparameters.get("proteomics_n_features", 1000)
            self.proteomics_normalization_width = hyperparameters.get("proteomics_normalization_width", 0.3)
            self.proteomics_normalization_downshift = hyperparameters.get("proteomics_normalization_downshift", 1.8)
            self.proteomics_transformer = ProteomicsMedianCenterAndImputeTransformer(
                feature_threshold=self.proteomics_feature_threshold,
                n_features=self.proteomics_n_features,
                normalization_downshift=self.proteomics_normalization_downshift,
                normalization_width=self.proteomics_normalization_width,
            )
        self.model = lgb.LGBMRegressor(
            n_estimators=hyperparameters.get("n_estimators", 100),
            learning_rate=hyperparameters.get("learning_rate", 0.1),
            max_depth=hyperparameters.get("max_depth", 6),
            num_leaves=hyperparameters.get("num_leaves", 63),
            subsample=hyperparameters.get("subsample", 0.8),
            colsample_bytree=hyperparameters.get("colsample_bytree", 0.8),
            reg_alpha=hyperparameters.get("reg_alpha", 0.0),
            reg_lambda=hyperparameters.get("reg_lambda", 0.0),
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: data path e.g. data/
        :param dataset_name: dataset name e.g. GDSC1
        :returns: FeatureDataset containing the cell line omics features
        """
        return load_multi_cell_line_view(self.cell_line_views, data_path, dataset_name, self.get_model_name())

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        Loads the drug features.

        :param data_path: path to the drug features, e.g., data/
        :param dataset_name: name of the dataset, e.g., GDSC1
        :returns: FeatureDataset containing the drug features
        """
        return load_single_drug_view(self.drug_views, data_path, dataset_name, self.get_model_name())

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "",
    ) -> None:
        """
        Trains the model.

        :param output: training dataset containing the response output
        :param cell_line_input: cell line omics features
        :param drug_input: drug features
        :param output_earlystopping: not used
        :param model_checkpoint_dir: not used
        """
        if "methylation" in self.cell_line_views:
            first_cl_feature = next(iter(cell_line_input.features.values()))
            n_met_features = first_cl_feature["methylation"].shape[0]
            n_components = min(self.pca_ncomp, n_met_features)
            self.methylation_pca = PCA(n_components=n_components)

        if "gene_expression" in self.cell_line_views or "methylation" in self.cell_line_views:
            cell_line_input = prepare_expression_and_methylation(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(output.cell_line_ids),
                training=True,
                gene_expression_scaler=self.gene_expression_scaler,
                methylation_scaler=self.methylation_scaler,
                methylation_pca=self.methylation_pca,
            )

        if "proteomics" in self.cell_line_views:
            cell_line_input = prepare_proteomics(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(output.cell_line_ids),
                training=True,
                transformer=self.proteomics_transformer,
            )

        inputs = self.get_feature_matrices(
            cell_line_ids=output.cell_line_ids,
            drug_ids=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

        array_list = [inputs[view] for view in self.cell_line_views + self.drug_views]
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
        :param cell_line_input: cell line omics features
        :param drug_input: drug features
        :returns: predicted response
        """
        if "gene_expression" in self.cell_line_views or "methylation" in self.cell_line_views:
            cell_line_input = prepare_expression_and_methylation(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(cell_line_ids),
                training=False,
                gene_expression_scaler=self.gene_expression_scaler,
                methylation_scaler=self.methylation_scaler,
                methylation_pca=self.methylation_pca,
            )

        if "proteomics" in self.cell_line_views:
            cell_line_input = prepare_proteomics(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(cell_line_ids),
                training=False,
                transformer=self.proteomics_transformer,
            )

        inputs = self.get_feature_matrices(
            cell_line_ids=cell_line_ids,
            drug_ids=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )

        array_list = [inputs[view] for view in self.cell_line_views + self.drug_views]
        x = np.concatenate(array_list, axis=1)
        return self.model.predict(x)

    def save(self, directory: str) -> None:
        """
        Saves the model to disk.

        :param directory: target directory
        """
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, os.path.join(directory, "model.pkl"))
        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(self.hyperparameters, f)
        if "gene_expression" in self.cell_line_views:
            joblib.dump(self.gene_expression_scaler, os.path.join(directory, "gene_scaler.pkl"))
        if "methylation" in self.cell_line_views:
            joblib.dump(self.methylation_scaler, os.path.join(directory, "methylation_scaler.pkl"))
            joblib.dump(self.methylation_pca, os.path.join(directory, "methylation_pca.pkl"))
        if self.proteomics_transformer is not None:
            joblib.dump(self.proteomics_transformer, os.path.join(directory, "proteomics_transformer.pkl"))

    @classmethod
    def load(cls, directory: str) -> "MultiViewLightGBM":
        """
        Loads the model from disk.

        :param directory: directory containing the saved model files
        :returns: restored MultiViewLightGBM instance
        """
        instance = cls()
        with open(os.path.join(directory, "hyperparameters.json")) as f:
            hyperparameters = json.load(f)
        instance.build_model(hyperparameters)
        instance.model = joblib.load(os.path.join(directory, "model.pkl"))
        if "gene_expression" in instance.cell_line_views:
            instance.gene_expression_scaler = joblib.load(os.path.join(directory, "gene_scaler.pkl"))
        if "methylation" in instance.cell_line_views:
            instance.methylation_scaler = joblib.load(os.path.join(directory, "methylation_scaler.pkl"))
            instance.methylation_pca = joblib.load(os.path.join(directory, "methylation_pca.pkl"))
        transformer_path = os.path.join(directory, "proteomics_transformer.pkl")
        if os.path.exists(transformer_path):
            instance.proteomics_transformer = joblib.load(transformer_path)
        return instance
