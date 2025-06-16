"""Contains sklearn baseline models: ElasticNet, RandomForest, SVM."""

import json
import os

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel

from ..utils import (
    ProteomicsMedianCenterAndImputeTransformer,
    load_and_select_gene_features,
    load_drug_fingerprint_features,
    prepare_proteomics,
    scale_gene_expression,
)


class SklearnModel(DRPModel):
    """Parent class that contains the common methods for the sklearn models."""

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]

    def __init__(self):
        """
        Initializes the model.

        Sets the model to None, which is initialized in the build_model method to the respective sklearn model.
        """
        super().__init__()
        self.model = None
        self.gene_expression_scaler = StandardScaler()
        self.proteomics_transformer = None

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

        :param hyperparameters: Custom hyperparameters for the model, have to be defined in the child class.
        """
        self.hyperparameters = hyperparameters

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
        :raises ValueError: If drug_input is None.
        """
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required for the sklearn models.")

        if "gene_expression" in self.cell_line_views:
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(output.cell_line_ids),
                training=True,
                gene_expression_scaler=self.gene_expression_scaler,
            )

        x = self.get_concatenated_features(
            cell_line_view=self.cell_line_views[0],
            drug_view=self.drug_views[0],
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
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

        :param drug_ids: drug ids
        :param cell_line_ids: cell line ids
        :param drug_input: drug input
        :param cell_line_input: cell line input
        :returns: predicted drug response
        :raises ValueError: If drug_input is None.
        """
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required.")
        if "gene_expression" in self.cell_line_views:
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(cell_line_ids),
                training=False,
                gene_expression_scaler=self.gene_expression_scaler,
            )

        x = self.get_concatenated_features(
            cell_line_view=self.cell_line_views[0],
            drug_view=self.drug_views[0],
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        return self.model.predict(x)

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: Path to the gene expression and landmark genes
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the cell line gene expression features, filtered through the landmark genes
        """
        return load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list="landmark_genes",
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        Load the drug features, in this case the fingerprints.

        :param data_path: Path to the data
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the drug fingerprints
        """
        return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)

    def save(self, directory: str) -> None:
        """
        Save the trained model and any associated preprocessing components to the given directory.

        Saves:
        - model.pkl: the trained sklearn model
        - hyperparameters.json: dictionary of model hyperparameters (if present)
        - scaler.pkl: fitted gene expression scaler (if present)
        - proteomics_transformer.pkl: fitted proteomics transformer (if present)

        :param directory: path to the directory where model files will be stored
        """
        os.makedirs(directory, exist_ok=True)
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
        if hyperparameters["l1_ratio"] == 0.0:
            self.model = Ridge(alpha=hyperparameters["alpha"])
        elif hyperparameters["l1_ratio"] == 1.0:
            self.model = Lasso(alpha=hyperparameters["alpha"])
        else:
            self.model = ElasticNet(
                alpha=hyperparameters["alpha"],
                l1_ratio=hyperparameters["l1_ratio"],
            )
        self.hyperparameters = hyperparameters


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
            and n_jobs.
        """
        if hyperparameters["max_depth"] == "None":
            hyperparameters["max_depth"] = None
        self.model = RandomForestRegressor(
            n_estimators=hyperparameters["n_estimators"],
            criterion=hyperparameters["criterion"],
            max_samples=hyperparameters["max_samples"],
            n_jobs=hyperparameters["n_jobs"],
        )
        self.hyperparameters = hyperparameters


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
        self.model = SVR(
            kernel=hyperparameters["kernel"],
            C=hyperparameters["C"],
            epsilon=hyperparameters["epsilon"],
            max_iter=hyperparameters["max_iter"],
        )
        self.hyperparameters = hyperparameters


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
        if hyperparameters["max_depth"] == "None":
            hyperparameters["max_depth"] = None
        self.model = HistGradientBoostingRegressor(
            max_iter=hyperparameters.get("max_iter", 100),
            learning_rate=hyperparameters.get("learning_rate", 0.1),
            max_depth=hyperparameters.get("max_depth", 3),
        )
        self.hyperparameters = hyperparameters


class ProteomicsRandomForest(RandomForest):
    """RandomForest model for drug response prediction using proteomics data."""

    cell_line_views = ["proteomics"]

    def __init__(self):
        """
        Initializes the model with specific hyperparameters.

        feature_threshold: for feature selection. Require that, e.g., 70% of the proteins are measured without NAs
        over all cell lines -> n_complete_features = number of proteins with at least 70% of the cell lines
        n_features: fallback for feature selection. Take top n complete features.
        Select max(n_complete_features, n_features) features.
        normalization_width: width of the Gaussian kernel for the median centering
        normalization_downshift: downshift of the median for the imputation of missing values
        """
        super().__init__()
        self.feature_threshold = 0.7
        self.n_features = 1000
        self.normalization_width = 0.3
        self.normalization_downshift = 1.8

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        :param hyperparameters: Hyperparameters for the model. Contains n_estimators, criterion, max_samples,
            and n_jobs.
        """
        super().build_model(hyperparameters)
        self.feature_threshold = hyperparameters.get("feature_threshold", 0.7)
        self.n_features = hyperparameters.get("n_features", 1000)
        self.normalization_width = hyperparameters.get("normalization_width", 0.3)
        self.normalization_downshift = hyperparameters.get("normalization_downshift", 1.8)
        self.proteomics_transformer = ProteomicsMedianCenterAndImputeTransformer(
            feature_threshold=self.feature_threshold,
            n_features=self.n_features,
            normalization_downshift=self.normalization_downshift,
            normalization_width=self.normalization_width,
        )

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: ProteomicsRandomForest
        """
        return "ProteomicsRandomForest"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: Path to the gene expression and landmark genes
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the cell line proteomics features, filtered through the landmark genes
        """
        return load_and_select_gene_features(
            feature_type="proteomics",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name,
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
        Trains the model.

        The number of features is the number of genes + the number of fingerprints.
        :param output: training dataset containing the response output
        :param cell_line_input: training dataset containing gene expression data
        :param drug_input: training dataset containing fingerprints data
        :param output_earlystopping: not needed
        :param model_checkpoint_dir: not needed
        :raises ValueError: If drug_input is None.
        """
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required for the sklearn models.")

        cell_line_input = prepare_proteomics(
            cell_line_input=cell_line_input,
            cell_line_ids=np.unique(output.cell_line_ids),
            training=True,
            transformer=self.proteomics_transformer,
        )

        x = self.get_concatenated_features(
            cell_line_view=self.cell_line_views[0],
            drug_view=self.drug_views[0],
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
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

        :param drug_ids: drug ids
        :param cell_line_ids: cell line ids
        :param drug_input: drug input
        :param cell_line_input: cell line input
        :returns: predicted drug response
        :raises ValueError: If drug_input is None.
        """
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required.")
        cell_line_input = prepare_proteomics(
            cell_line_input=cell_line_input,
            cell_line_ids=np.unique(cell_line_ids),
            training=False,
            transformer=self.proteomics_transformer,
        )
        if self.model is None:
            print("No training data was available, or model not trained predicting NA.")
            return np.array([np.nan] * len(cell_line_ids))
        x = self.get_concatenated_features(
            cell_line_view=self.cell_line_views[0],
            drug_view=self.drug_views[0],
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        return self.model.predict(x)


class ProteomicsElasticNetModel(ElasticNetModel):
    """ElasticNet model for drug response prediction using proteomics data."""

    cell_line_views = ["proteomics"]

    def __init__(self):
        """
        Initializes the model with specific hyperparameters.

        feature_threshold: for feature selection. Require that, e.g., 70% of the proteins are measured without NAs
        over all cell lines -> n_complete_features = number of proteins with at least 70% of the cell lines
        n_features: fallback for feature selection. Take top n complete features.
        Select max(n_complete_features, n_features) features.
        normalization_width: width of the Gaussian kernel for the median centering
        normalization_downshift: downshift of the median for the imputation of missing values
        """
        super().__init__()
        self.feature_threshold = 0.7
        self.n_features = 1000
        self.normalization_width = 0.3
        self.normalization_downshift = 1.8

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters.

        :param hyperparameters: Hyperparameters for the model.
        """
        super().build_model(hyperparameters)
        self.feature_threshold = hyperparameters.get("feature_threshold", 0.7)
        self.n_features = hyperparameters.get("n_features", 1000)
        self.normalization_width = hyperparameters.get("normalization_width", 0.3)
        self.normalization_downshift = hyperparameters.get("normalization_downshift", 1.8)
        self.proteomics_transformer = ProteomicsMedianCenterAndImputeTransformer(
            feature_threshold=self.feature_threshold,
            n_features=self.n_features,
            normalization_downshift=self.normalization_downshift,
            normalization_width=self.normalization_width,
        )

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: ProteomicsElasticNet
        """
        return "ProteomicsElasticNet"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: Path to the gene expression and landmark genes
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the cell line proteomics features, filtered through the landmark genes
        """
        return load_and_select_gene_features(
            feature_type="proteomics",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name,
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
        Trains the model.

        The number of features is the number of genes + the number of fingerprints.
        :param output: training dataset containing the response output
        :param cell_line_input: training dataset containing gene expression data
        :param drug_input: training dataset containing fingerprints data
        :param output_earlystopping: not needed
        :param model_checkpoint_dir: not needed
        :raises ValueError: If drug_input is None.
        """
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required for the sklearn models.")

        cell_line_input = prepare_proteomics(
            cell_line_input=cell_line_input,
            cell_line_ids=np.unique(output.cell_line_ids),
            training=True,
            transformer=self.proteomics_transformer,
        )
        x = self.get_concatenated_features(
            cell_line_view=self.cell_line_views[0],
            drug_view=self.drug_views[0],
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
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

        :param drug_ids: drug ids
        :param cell_line_ids: cell line ids
        :param drug_input: drug input
        :param cell_line_input: cell line input
        :returns: predicted drug response
        :raises ValueError: If drug_input is None.
        """
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required.")
        cell_line_input = prepare_proteomics(
            cell_line_input=cell_line_input,
            cell_line_ids=np.unique(cell_line_ids),
            training=False,
            transformer=self.proteomics_transformer,
        )
        if self.model is None:
            print("No training data was available, or model not trained predicting NA.")
            return np.array([np.nan] * len(cell_line_ids))
        x = self.get_concatenated_features(
            cell_line_view=self.cell_line_views[0],
            drug_view=self.drug_views[0],
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        return self.model.predict(x)
