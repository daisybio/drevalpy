"""Contains sklearn baseline models: ElasticNet, RandomForest, SVM, AdaBoost."""


import numpy as np

from drevalpy.components.register_builtins import ensure_components_registered
from drevalpy.data.features import _get_view_as_list, load_single_cell_line_view, load_single_drug_view
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models._component_bridge import (
    ComponentDRPBridge,
    preview_sklearn_estimator,
    save_component_stack,
)
from drevalpy.models._legacy_checkpoint_loaders import (
    has_component_stack,
    load_legacy_sklearn_checkpoint,
    load_native_checkpoint,
)
from drevalpy.models._legacy_state_accessors import (
    sklearn_estimator_from_bridge,
    sklearn_featurizer_state_from_bridge,
)
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.factory import SKLEARN_PREDICTOR_BY_MODEL_NAME, sklearn_model_config


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
        self._component_bridge = ComponentDRPBridge()
        self._preview_model = None
        # proteomics-specific defaults
        self.proteomics_feature_threshold = 0.7
        self.proteomics_n_features = 1000
        self.proteomics_normalization_width = 0.3
        self.proteomics_normalization_downshift = 1.8
        self.methylation_n_components = 100

    @property
    def model(self):
        fitted = sklearn_estimator_from_bridge(self._component_bridge)
        if fitted is not None:
            return fitted
        return self._preview_model

    @property
    def gene_expression_scaler(self):
        return sklearn_featurizer_state_from_bridge(self._component_bridge).get("gene_expression_scaler")

    @property
    def methylation_scaler(self):
        return sklearn_featurizer_state_from_bridge(self._component_bridge).get("methylation_scaler")

    @property
    def methylation_pca(self):
        return sklearn_featurizer_state_from_bridge(self._component_bridge).get("methylation_pca")

    @property
    def proteomics_transformer(self):
        return sklearn_featurizer_state_from_bridge(self._component_bridge).get("proteomics_transformer")

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

        Flexible input support: Initializes the cell_line_views and drug_views from the
        public hyperparameter dict passed to build_model(). If nothing is specified,
        gene_expression and fingerprints are used.

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
        self._init_component_model()

    def _normalize_hyperparameters(self, hyperparameters: dict) -> dict:
        normalized = dict(hyperparameters)
        if normalized.get("max_depth") == "None":
            normalized["max_depth"] = None
        return normalized

    def _init_component_model(self) -> None:
        ensure_components_registered()
        predictor_type = SKLEARN_PREDICTOR_BY_MODEL_NAME[self.get_model_name()]
        hp = self._normalize_hyperparameters(self.hyperparameters)
        hp["cell_line_views"] = self.cell_line_views
        hp["drug_views"] = self.drug_views
        config = sklearn_model_config(predictor_type, hp)
        self._component_bridge.set_composed_config(config)
        self._preview_model = preview_sklearn_estimator(self._component_bridge, hp)

    def _init_proteomics_features(self, hyperparameters: dict):
        self.proteomics_feature_threshold = hyperparameters.get("proteomics_feature_threshold", 0.7)
        self.proteomics_n_features = hyperparameters.get("proteomics_n_features", 1000)
        self.proteomics_normalization_width = hyperparameters.get("proteomics_normalization_width", 0.3)
        self.proteomics_normalization_downshift = hyperparameters.get("proteomics_normalization_downshift", 1.8)

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
        if len(output) == 0:
            print("No training data provided, will predict NA.")
            self._preview_model = None
            return
        self._component_bridge.train(output, cell_line_input, drug_input)

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
        if not self._component_bridge.is_trained():
            print("No training data was available, predicting NA.")
            return np.array([np.nan] * len(cell_line_ids))
        return self._component_bridge.predict(cell_line_ids, drug_ids, cell_line_input, drug_input)

    def save(self, directory: str) -> None:
        """
        Save the trained model and any associated preprocessing components to the given directory.

        Persists the composed component stack when trained; otherwise raises if no model exists.

        :param directory: path to the directory where model files will be stored
        :raises ValueError: if the model is not trained
        """
        if not self._component_bridge.is_trained():
            raise ValueError("Cannot save: model is not trained.")
        save_component_stack(
            self._component_bridge,
            directory,
            hyperparameters=getattr(self, "hyperparameters", {}),
        )

    @classmethod
    def load(cls, directory: str) -> "SklearnModel":
        """
        Load a trained sklearn-based model and its preprocessing components from disk.

        :param directory: path to the directory where model files are stored
        :return: an instance of the model with restored state
        :raises FileNotFoundError: if no recognized model artifacts are present
        """
        if has_component_stack(directory):
            instance = cls()
            load_native_checkpoint(instance, directory)
            return instance

        instance = cls()
        load_legacy_sklearn_checkpoint(instance, directory)
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
