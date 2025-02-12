"""SingleDrugElasticNet and SingleDrugProteomicsElasticNet classes. Fit an Elastic net for each drug seperately."""

import numpy as np
from sklearn.linear_model import ElasticNet

from ...datasets.dataset import DrugResponseDataset, FeatureDataset
from ..utils import load_and_reduce_gene_features
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
        :raises ValueError: if drug_input is not None
        """
        if drug_input is not None:
            raise ValueError("SingleDrugElasticNet does not support drug_input!")

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
        Load drug features. Not needed for SingleDrugElasticNet.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: None
        """
        return None


class SingleDrugProteomicsElasticNet(SingleDrugElasticNet):
    """SingleDrugProteomicsElasticNet class."""

    cell_line_views = ["proteomics"]
    is_single_drug_model = True

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
        return load_and_reduce_gene_features(
            feature_type="proteomics",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path, dataset_name):
        """
        Load drug features. Not needed for SingleDrugProteomicsElasticNet.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: None
        """
        return None

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
            raise ValueError("SingleDrugElasticNet does not support drug_input!")

        if len(output) > 0:
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
        x = self.get_concatenated_features(
            cell_line_view="proteomics",
            drug_view=None,
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=None,
        )
        return self.model.predict(x)
