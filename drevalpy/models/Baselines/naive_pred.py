from typing import Dict
import numpy as np
from numpy.typing import ArrayLike

from drevalpy.datasets.dataset import FeatureDataset, DrugResponseDataset
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.utils import load_cl_ids_from_csv, load_drug_ids_from_csv, unique


class NaivePredictor(DRPModel):
    model_name = "NaivePredictor"
    cell_line_views = ["cell_line_id"]
    drug_views = ["drug_id"]

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.dataset_mean = None

    def build_model(self, hyperparameters: Dict, *args, **kwargs):
        pass

    def train(self, output: DrugResponseDataset, *args, **kwargs) -> None:
        """
        Computes the overall mean of the output response values and saves them.
        :param output: training dataset containing the response output
        """
        self.dataset_mean = np.mean(output.response)

    def predict(self, cell_line_ids: np.ndarray = None, *args, **kwargs) -> np.ndarray:
        """
        Predicts the dataset mean for each drug-cell line combination
        :param cell_line_ids: cell line ids
        :return: array of the same length as the input cell line id containing the dataset mean
        """
        return np.full(cell_line_ids.shape[0], self.dataset_mean)

    def save(self, path):
        raise NotImplementedError("Naive predictor does not support saving yet ...")

    def load(self, path):
        raise NotImplementedError("Naive predictor does not support loading yet ...")

    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
        return load_cl_ids_from_csv(data_path, dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        return load_drug_ids_from_csv(data_path, dataset_name)


class NaiveDrugMeanPredictor(DRPModel):
    model_name = "NaiveDrugMeanPredictor"
    cell_line_views = ["cell_line_id"]
    drug_views = ["drug_id"]

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.drug_means = None
        self.dataset_mean = None

    def build_model(self, *args, **kwargs):
        pass

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset = None,
        *args,
        **kwargs
    ) -> None:
        """
        Computes the mean per drug. If - later on - the drug is not in the training set, the overall mean is used.
        :param output: training dataset containing the response output
        :param drug_input: drug id
        """
        drug_ids = drug_input.get_feature_matrix(
            view="drug_id", identifiers=output.drug_ids
        )
        self.dataset_mean = np.mean(output.response)
        self.drug_means = {}

        for drug_response, drug_feature in zip(
            unique(output.drug_ids), unique(drug_ids)
        ):
            responses_drug = output.response[drug_feature == output.drug_ids]
            if len(responses_drug) > 0:
                # prevent nan response
                self.drug_means[drug_response] = np.mean(responses_drug)

    def predict(self, drug_ids: ArrayLike, *args, **kwargs) -> np.ndarray:
        """
        Predicts the drug mean for each drug-cell line combination. If the drug is not in the training set, the dataset mean is used.
        :param cell_line_id: cell line ids
        :param drug_id: drug ids
        :return: array of the same length as the input drug_id containing the drug mean
        """
        return np.array([self.predict_drug(drug) for drug in drug_ids])

    def predict_drug(self, drug_id: str):
        if drug_id in self.drug_means:
            return self.drug_means[drug_id]
        return self.dataset_mean

    def save(self, path):
        raise NotImplementedError("Naive predictor does not support saving yet ...")

    def load(self, path):
        raise NotImplementedError("Naive predictor does not support loading yet ...")

    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
        return load_cl_ids_from_csv(data_path, dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        return load_drug_ids_from_csv(data_path, dataset_name)


class NaiveCellLineMeanPredictor(DRPModel):
    model_name = "NaiveCellLineMeanPredictor"
    cell_line_views = ["cell_line_id"]
    drug_views = ["drug_id"]

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.cell_line_means = None
        self.dataset_mean = None

    def build_model(self, *args, **kwargs):
        pass

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        *args,
        **kwargs
    ) -> None:
        """
        Computes the mean per cell line. If - later on - the cell line is not in the training set, the overall mean is used.
        :param output: training dataset containing the response output
        :param cell_line_input: cell line inputs
        :param drug_input: drug inputs
        """
        cell_line_ids = cell_line_input.get_feature_matrix(
            view="cell_line_id", identifiers=output.cell_line_ids
        )
        self.dataset_mean = np.mean(output.response)
        self.cell_line_means = {}

        for cell_line_response, cell_line_feature in zip(
            unique(output.cell_line_ids), unique(cell_line_ids)
        ):
            responses_cl = output.response[cell_line_feature == output.cell_line_ids]
            if len(responses_cl) > 0:
                # prevent nan response
                self.cell_line_means[cell_line_response] = np.mean(responses_cl)

    def predict(self, cell_line_ids: ArrayLike, *args, **kwargs) -> np.ndarray:
        """
        Predicts the cell line mean for each drug-cell line combination. If the cell line is not in the training set, the dataset mean is used.
        :param cell_line_ids: cell line ids
        :return: array of the same length as the input cell_line_id containing the cell line mean
        """
        return np.array([self.predict_cl(cl) for cl in cell_line_ids])

    def predict_cl(self, cl_id: str):
        if cl_id in self.cell_line_means:
            return self.cell_line_means[cl_id]
        return self.dataset_mean

    def save(self, path):
        raise NotImplementedError("Naive predictor does not support saving yet ...")

    def load(self, path):
        raise NotImplementedError("Naive predictor does not support loading yet ...")

    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
        return load_cl_ids_from_csv(data_path, dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        return load_drug_ids_from_csv(data_path, dataset_name)
