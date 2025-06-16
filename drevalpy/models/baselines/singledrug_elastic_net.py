"""SingleDrugElasticNet and SingleDrugProteomicsElasticNet classes. Fit an Elastic net for each drug separately."""

import json
import os

import joblib
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..utils import (
    ProteomicsMedianCenterAndImputeTransformer,
    load_and_select_gene_features,
    prepare_proteomics,
    scale_gene_expression,
)
from .sklearn_models import SklearnModel


class SingleDrugElasticNet(SklearnModel):
    """SingleDrugElasticNet class."""

    is_single_drug_model = True
    drug_views = []
    cell_line_views = ["gene_expression"]
    early_stopping = False

    def build_model(self, hyperparameters):
        """
        Builds the model from hyperparameters.

        :param hyperparameters: Elastic net hyperparameters
        """
        self.model = ElasticNet(**hyperparameters)
        self.gene_expression_scaler = StandardScaler()

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SingleDrugElasticNet
        """
        return "SingleDrugElasticNet"

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Trains the model; the number of features is the number of fingerprints.

        :param output: training dataset containing the response output
        :param cell_line_input: training dataset containing gene expression data
        :param drug_input: not needed
        :param output_earlystopping: not needed
        :param model_checkpoint_dir: not needed as checkpoints are not saved
        """
        if len(output) > 0:
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(output.cell_line_ids),
                training=True,
                gene_expression_scaler=self.gene_expression_scaler,
            )
            x = self.get_concatenated_features(
                cell_line_view="gene_expression",
                drug_view=None,
                cell_line_ids_output=output.cell_line_ids,
                drug_ids_output=output.drug_ids,
                cell_line_input=cell_line_input,
                drug_input=None,
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
        Predicts the drug response for the given cell lines.

        :param cell_line_ids: cell line ids
        :param drug_ids: drug ids, not needed here
        :param cell_line_input: cell line input
        :param drug_input: drug input, not needed here
        :returns: predicted drug response
        :raises ValueError: if drug_input is not None
        """
        if drug_input is not None:
            raise ValueError("drug_input is not needed for SingleDrugModel.")

        if self.model is None:
            print("No training data was available, predicting NA.")
            return np.array([np.nan] * len(cell_line_ids))
        cell_line_input = scale_gene_expression(
            cell_line_input=cell_line_input,
            cell_line_ids=np.unique(cell_line_ids),
            training=False,
            gene_expression_scaler=self.gene_expression_scaler,
        )
        x = self.get_concatenated_features(
            cell_line_view="gene_expression",
            drug_view=None,
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=None,
        )
        return self.model.predict(x)

    def load_drug_features(self, data_path, dataset_name):
        """
        Load drug features. Not needed for SingleDrugElasticNet.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: None
        """
        return None


class SingleDrugProteomicsElasticNet(SingleDrugElasticNet):
    """SingleDrugProteomicsElasticNet class."""

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
        hyperparameters = hyperparameters.copy()
        self.feature_threshold = hyperparameters.pop("feature_threshold", 0.7)
        self.n_features = hyperparameters.pop("n_features", 1000)
        self.normalization_width = hyperparameters.pop("normalization_width", 0.3)
        self.normalization_downshift = hyperparameters.pop("normalization_downshift", 1.8)
        self.proteomics_transformer = ProteomicsMedianCenterAndImputeTransformer(
            feature_threshold=self.feature_threshold,
            n_features=self.n_features,
            normalization_downshift=self.normalization_downshift,
            normalization_width=self.normalization_width,
        )
        super().build_model(hyperparameters)

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SingleDrugProteomicsElasticNet
        """
        return "SingleDrugProteomicsElasticNet"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the proteomics data.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: proteomics data
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
        Trains the model; the number of features is the number of fingerprints.

        :param output: training dataset containing the response output
        :param cell_line_input: training dataset containing gene expression data
        :param drug_input: not needed
        :param output_earlystopping: not needed
        :param model_checkpoint_dir: not needed as checkpoints are not saved
        """
        if len(output) > 0:
            # log transform
            cell_line_input = prepare_proteomics(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(output.cell_line_ids),
                training=True,
                transformer=self.proteomics_transformer,
            )
            x = self.get_concatenated_features(
                cell_line_view="proteomics",
                drug_view=None,
                cell_line_ids_output=output.cell_line_ids,
                drug_ids_output=output.drug_ids,
                cell_line_input=cell_line_input,
                drug_input=None,
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
        Predicts the drug response for the given cell lines.

        :param cell_line_ids: cell line ids
        :param drug_ids: drug ids, not needed here
        :param cell_line_input: cell line input
        :param drug_input: drug input, not needed here
        :returns: predicted drug response
        :raises ValueError: if drug_input is not None
        """
        if drug_input is not None:
            raise ValueError("drug_input is not needed.")

        if self.model is None:
            print("No training data was available, predicting NA.")
            return np.array([np.nan] * len(cell_line_ids))

        cell_line_input = prepare_proteomics(
            cell_line_input=cell_line_input,
            cell_line_ids=np.unique(cell_line_ids),
            training=False,
            transformer=self.proteomics_transformer,
        )

        x = self.get_concatenated_features(
            cell_line_view="proteomics",
            drug_view=None,
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=None,
        )
        return self.model.predict(x)

    def save(self, directory: str) -> None:
        """
        Save the trained model and proteomics transformer.

        Saves:
        - model.pkl: the fitted ElasticNet model
        - transformer.pkl: the fitted ProteomicsMedianCenterAndImputeTransformer

        :param directory: Target directory for saving model files
        :raises ValueError: If model or transformer is not initialized
        """
        os.makedirs(directory, exist_ok=True)

        if self.model is None:
            raise ValueError("Cannot save: model is not trained.")

        joblib.dump(self.model, os.path.join(directory, "elasticnet_model.pkl"))
        joblib.dump(self.proteomics_transformer, os.path.join(directory, "proteomics_transformer.pkl"))

        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(
                {
                    "feature_threshold": self.feature_threshold,
                    "n_features": self.n_features,
                    "normalization_width": self.normalization_width,
                    "normalization_downshift": self.normalization_downshift,
                },
                f,
            )

    @classmethod
    def load(cls, directory: str) -> "SingleDrugProteomicsElasticNet":
        """
        Load a trained SingleDrugProteomicsElasticNet model and transformer.

        Loads:
        - model.pkl: trained ElasticNet model
        - transformer.pkl: fitted ProteomicsMedianCenterAndImputeTransformer

        :param directory: Directory where the model files are stored
        :return: Loaded instance of SingleDrugProteomicsElasticNet
        """
        instance = cls()

        with open(os.path.join(directory, "hyperparameters.json")) as f:
            hyperparameters = json.load(f)
        instance.build_model(hyperparameters)

        instance.model = joblib.load(os.path.join(directory, "elasticnet_model.pkl"))
        instance.proteomics_transformer = joblib.load(os.path.join(directory, "proteomics_transformer.pkl"))

        return instance
