"""
Contains the SingleDrugRandomForest class.

It is a RandomForest model that uses only gene expression dataset for drug response prediction and trains one model
per drug.
"""

import numpy as np

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..utils import ProteomicsMedianCenterAndImputeTransformer, load_and_select_gene_features, prepare_proteomics
from .sklearn_models import RandomForest


class SingleDrugRandomForest(RandomForest):
    """SingleDrugRandomForest class."""

    is_single_drug_model = True
    drug_views = []
    early_stopping = False

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SingleDrugRandomForest
        """
        return "SingleDrugRandomForest"

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
        :raises ValueError: if drug_input is not None
        """
        if drug_input is not None:
            raise ValueError("SingleDrugRandomForest does not support drug_input!")

        if len(output) > 0:
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
            raise ValueError("drug_input is not needed.")

        if self.model is None:
            print("No training data was available, predicting NA.")
            return np.array([np.nan] * len(cell_line_ids))
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
        Load drug features. Not needed for SingleDrugRandomForest.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: None
        """
        return None


class SingleDrugProteomicsRandomForest(SingleDrugRandomForest):
    """SingleDrugProteomicsRandomForest class."""

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

        :returns: SingleDrugProteomicsRandomForest
        """
        return "SingleDrugProteomicsRandomForest"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the proteomics features.

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
