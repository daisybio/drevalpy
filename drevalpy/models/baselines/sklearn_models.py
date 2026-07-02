"""Contains sklearn baseline models: ElasticNet, RandomForest, SVM, AdaBoost."""

import json
import os

import joblib
import numpy as np
from sklearn.ensemble import AdaBoostRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel

from ..utils import (
    ProteomicsMedianCenterAndImputeTransformer,
    _get_view_as_list,
    load_single_cell_line_view,
    load_single_drug_view,
    prepare_proteomics,
    scale_gene_expression,
)


class SklearnModel(DRPModel):
    """Parent class that contains the common methods for the sklearn models."""

    cell_line_views = []
    drug_views = []

    def __init__(self):
        """
        Initializes the model.

        Sets the model to None, which is initialized in the build_model method to the respective sklearn model.

        Initializes omic-specific defaults:
        *  For gene expression, a StandardScaler is initialized which will standardize the gene expression data.
        *  For proteomics, default parameters for the ProteomicsMedianCenterAndImputeTransformer are initialized
        (feature_threshold=0.7, n_features=1000, normalization_width=0.3, normalization_downshift=1.8).
        """
        super().__init__()
        self.model = None
        self.gene_expression_scaler = StandardScaler()
        # proteomics-specific defaults
        self.proteomics_transformer = None
        self.proteomics_feature_threshold = 0.7
        self.proteomics_n_features = 1000
        self.proteomics_normalization_width = 0.3
        self.proteomics_normalization_downshift = 1.8
        # methylation-specific defaults
        self.methylation_pca = None
        self.methylation_n_components = 100

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :raises NotImplementedError: If the method is not implemented in the child class.
        """
        raise NotImplementedError("get_model_name method has to be implemented in the child class.")

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        Flexible input support: Initializes the cell_line_views and drug_views to the values specified in the
        hyperparameters.yaml file. If nothing is specified, gene_expression and fingerprints are used.

        If proteomics is specified in the hyperparameters, the ProteomicsMedianCenterAndImputeTransformer
        is initialized.

        :param hyperparameters: Custom hyperparameters for the model, have to be defined in the child class.
        """
        # Log hyperparameters to wandb if enabled
        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = hyperparameters
        self.cell_line_views = _get_view_as_list(hyperparameters.get("cell_line_views", ["gene_expression"]))
        self.drug_views = _get_view_as_list(hyperparameters.get("drug_views", ["fingerprints"]))

        # proteomics features are not supported for all models
        if "proteomics" in self.cell_line_views:
            self._init_proteomics_features(hyperparameters)

        # methylation features are not supported for all models
        if "methylation" in self.cell_line_views:
            self.methylation_n_components = hyperparameters.get("methylation_n_components", 100)

    def _init_proteomics_features(self, hyperparameters: dict):
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

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features for a single-view sklearn model.

        :param data_path: Path to the data
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the cell line features
        """
        return load_single_cell_line_view(self.cell_line_views, data_path, dataset_name, self.get_model_name())

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        Load the drug features for a single-view sklearn model.

        :param data_path: Path to the data
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the drug features
        """
        return load_single_drug_view(self.drug_views, data_path, dataset_name, self.get_model_name())

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Trains the model.

        The number of features is the number of genes + the number of fingerprints.
        :param output: training dataset containing the response output
        :param cell_line_input: training dataset containing gene expression data
        :param drug_input: training dataset containing fingerprints data
        :param output_earlystopping: not needed
        :param model_checkpoint_dir: not needed
        """
        if len(output) > 0:
            if "gene_expression" in self.cell_line_views:
                cell_line_input = scale_gene_expression(
                    cell_line_input=cell_line_input,
                    cell_line_ids=np.unique(output.cell_line_ids),
                    training=True,
                    gene_expression_scaler=self.gene_expression_scaler,
                )
            elif "proteomics" in self.cell_line_views:
                cell_line_input = prepare_proteomics(
                    cell_line_input=cell_line_input,
                    cell_line_ids=np.unique(output.cell_line_ids),
                    training=True,
                    transformer=self.proteomics_transformer,
                )
            if len(self.drug_views) == 0:
                # support for single-drug models
                drug_view = None
            else:
                drug_view = self.drug_views[0]

            x = self.get_concatenated_features(
                cell_line_view=self.cell_line_views[0],
                drug_view=drug_view,
                cell_line_ids_output=output.cell_line_ids,
                drug_ids_output=output.drug_ids,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
            )
            self.model.fit(x, output.response)
        else:
            print("No training data provided, will predict NA.")
            self.model = None

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the response for the given input.

        :param drug_ids: drug ids
        :param cell_line_ids: cell line ids
        :param drug_input: drug input
        :param cell_line_input: cell line input
        :returns: predicted drug response
        """
        if self.model is None:
            print("No training data was available, predicting NA.")
            return np.array([np.nan] * len(cell_line_ids))

        if "gene_expression" in self.cell_line_views:
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(cell_line_ids),
                training=False,
                gene_expression_scaler=self.gene_expression_scaler,
            )
        elif "proteomics" in self.cell_line_views:
            cell_line_input = prepare_proteomics(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(cell_line_ids),
                training=False,
                transformer=self.proteomics_transformer,
            )

        if len(self.drug_views) == 0:
            # support for single-drug models
            drug_view = None
        else:
            drug_view = self.drug_views[0]

        x = self.get_concatenated_features(
            cell_line_view=self.cell_line_views[0],
            drug_view=drug_view,
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        return self.model.predict(x)

    def save(self, directory: str) -> None:
        """
        Save the trained model and any associated preprocessing components to the given directory.

        Saves:
        - model.pkl: the trained sklearn model
        - hyperparameters.json: dictionary of model hyperparameters (if present)
        - scaler.pkl: fitted gene expression scaler (if present)
        - proteomics_transformer.pkl: fitted proteomics transformer (if present)

        :param directory: path to the directory where model files will be stored
        :raises ValueError: if the model is not trained
        """
        os.makedirs(directory, exist_ok=True)
        if self.model is None:
            raise ValueError("Cannot save: model is not trained.")

        joblib.dump(self.model, os.path.join(directory, "model.pkl"))
        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(getattr(self, "hyperparameters", {}), f)
        if self.gene_expression_scaler is not None:
            joblib.dump(self.gene_expression_scaler, os.path.join(directory, "scaler.pkl"))
        if self.proteomics_transformer is not None:
            joblib.dump(self.proteomics_transformer, os.path.join(directory, "proteomics_transformer.pkl"))

    @classmethod
    def load(cls, directory: str) -> "SklearnModel":
        """
        Load a trained sklearn-based model and its preprocessing components from disk.

        Loads:
        - model.pkl: the trained sklearn model
        - hyperparameters.json: model hyperparameters (optional)
        - scaler.pkl: gene expression scaler (optional)
        - proteomics_transformer.pkl: proteomics transformer (optional)

        :param directory: path to the directory where model files are stored
        :return: an instance of the model with restored state
        :raises FileNotFoundError: if model.pkl is missing
        """
        model_path = os.path.join(directory, "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found")

        instance = cls()

        hyperparams_path = os.path.join(directory, "hyperparameters.json")
        with open(hyperparams_path) as f:
            hyperparameters = json.load(f)
        instance.build_model(hyperparameters)
        instance.model = joblib.load(model_path)

        scaler_path = os.path.join(directory, "scaler.pkl")
        if os.path.exists(scaler_path):
            instance.gene_expression_scaler = joblib.load(scaler_path)

        transformer_path = os.path.join(directory, "proteomics_transformer.pkl")
        if os.path.exists(transformer_path):
            instance.proteomics_transformer = joblib.load(transformer_path)

        return instance


class ElasticNetModel(SklearnModel):
    """ElasticNet model for drug response prediction."""

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: ElasticNet
        """
        return "ElasticNet"

    def build_model(self, hyperparameters: dict):
        """
        Builds the ElasticNet model from hyperparameters.

        :param hyperparameters: Contains L1 ratio and alpha.
        """
        super().build_model(hyperparameters)
        if self.hyperparameters["l1_ratio"] == 0.0:
            self.model = Ridge(alpha=self.hyperparameters["alpha"])
        elif self.hyperparameters["l1_ratio"] == 1.0:
            self.model = Lasso(alpha=self.hyperparameters["alpha"])
        else:
            self.model = ElasticNet(
                alpha=self.hyperparameters["alpha"],
                l1_ratio=self.hyperparameters["l1_ratio"],
            )


class RandomForest(SklearnModel):
    """RandomForest model for drug response prediction."""

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: RandomForest
        """
        return "RandomForest"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        :param hyperparameters: Hyperparameters for the model. Contains n_estimators, criterion, max_samples,
            max_depth and n_jobs.
        """
        super().build_model(hyperparameters)
        if self.hyperparameters["max_depth"] == "None":
            self.hyperparameters["max_depth"] = None
        self.model = RandomForestRegressor(
            n_estimators=self.hyperparameters["n_estimators"],
            criterion=self.hyperparameters["criterion"],
            max_samples=self.hyperparameters["max_samples"],
            max_depth=self.hyperparameters["max_depth"],
            n_jobs=self.hyperparameters["n_jobs"],
        )


class SVMRegressor(SklearnModel):
    """SVM model for drug response prediction."""

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SVR (Support Vector Regressor)
        """
        return "SVR"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        :param hyperparameters: Hyperparameters for the model. Contains kernel, C, epsilon, and max_iter.
        """
        super().build_model(hyperparameters)
        self.model = SVR(
            kernel=self.hyperparameters["kernel"],
            C=self.hyperparameters["C"],
            epsilon=self.hyperparameters["epsilon"],
            max_iter=self.hyperparameters["max_iter"],
        )


class GradientBoosting(SklearnModel):
    """Gradient Boosting model for drug response prediction."""

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: GradientBoosting
        """
        return "GradientBoosting"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        :param hyperparameters: Hyperparameters for the model. Contains n_estimators, learning_rate, max_depth,
            and subsample
        """
        super().build_model(hyperparameters)
        if self.hyperparameters["max_depth"] == "None":
            self.hyperparameters["max_depth"] = None
        self.model = HistGradientBoostingRegressor(
            max_iter=self.hyperparameters.get("max_iter", 100),
            learning_rate=self.hyperparameters.get("learning_rate", 0.1),
            max_depth=self.hyperparameters.get("max_depth", 3),
        )


class AdaBoostDecisionTree(SklearnModel):
    """AdaBoost model using Decision Trees as week learners for drug response prediction."""

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: AdaBoostDecisionTree
        """
        return "AdaBoostDecisionTree"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        :param hyperparameters: Hyperparameters for the model. Contains n_estimators, max_depth,
            min_samples_split and min_samples_leaf.
        """
        super().build_model(hyperparameters)
        self.model = AdaBoostRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=self.hyperparameters["max_depth"],
                min_samples_split=self.hyperparameters["min_samples_split"],
                min_samples_leaf=self.hyperparameters["min_samples_leaf"],
            ),
            n_estimators=self.hyperparameters["n_estimators"],
        )


class LassoModel(SklearnModel):
    """Lasso regression model for drug response prediction."""

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: Lasso
        """
        return "Lasso"

    def build_model(self, hyperparameters: dict):
        """
        Builds the Lasso model from hyperparameters.

        :param hyperparameters: Contains alpha.
        """
        super().build_model(hyperparameters)
        self.model = Lasso(
            alpha=self.hyperparameters["alpha"],
            max_iter=10000,
            tol=1e-3,
            selection="random",
        )


class KNNRegressor(SklearnModel):
    """KNNRegressor model for using k-nearest neighbors for drug response prediction."""

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: KNNRegressor
        """
        return "KNNRegressor"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        :param hyperparameters: Hyperparameters for the model. Contains neighbors, weights.
        """
        super().build_model(hyperparameters)
        self.model = KNeighborsRegressor(
            n_neighbors=self.hyperparameters["n_neighbors"], weights=self.hyperparameters.get("weights", "distance")
        )
