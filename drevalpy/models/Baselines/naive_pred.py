from typing import Dict, List
import numpy as np

from drevalpy.datasets.dataset import FeatureDataset, DrugResponseDataset
from drevalpy.models.drp_model import DRPModel
from ..utils import load_cl_ids_from_csv, load_drug_ids_from_csv


class NaivePredictor(DRPModel):
    model_name = "NaivePredictor"
    cell_line_views = ["cell_line_id"]
    drug_views = ["drug_id"]

    def __init__(self, target, *args, **kwargs):
        super().__init__(target, args, kwargs)
        self.dataset_mean = None

    def build_model(self, hyperparameters: Dict, *args, **kwargs):
        pass

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_id: np.ndarray = None,
        drug_id: np.ndarray = None,
    ) -> None:
        """
        Computes the overall mean of the output response values and saves them.
        :param output: training dataset containing the response output
        :param cell_line_id: cell line id
        :param drug_id: drug id
        """
        self.dataset_mean = np.mean(output.response)

    def predict(
        self, cell_line_id: np.ndarray = None, drug_id: np.ndarray = None
    ) -> np.ndarray:
        """
        Predicts the dataset mean for each drug-cell line combination
        :param cell_line_id: cell line id
        :param drug_id: drug id
        :return: array of the same length as the input cell line id containing the dataset mean
        """
        return np.full(cell_line_id.shape[0], self.dataset_mean)

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

    def __init__(self, target, *args, **kwargs):
        super().__init__(target, args, kwargs)
        self.drug_means = None
        self.dataset_mean = None

    def build_model(self, *args, **kwargs):
        pass

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_id: np.ndarray = None,
        drug_id: np.ndarray = None,
    ) -> None:
        """
        Computes the mean per drug. If - later on - the drug is not in the training set, the overall mean is used.
        :param output: training dataset containing the response output
        :param cell_line_id: cell line id
        :param drug_id: drug id
        """
        self.dataset_mean = np.mean(output.response)
        self.drug_means = {}
        for drug in np.unique(drug_id):
            self.drug_means[drug] = np.mean(output.response[drug_id == drug])

    def predict(
        self, cell_line_id: np.ndarray = None, drug_id: np.ndarray = None
    ) -> np.ndarray:
        """
        Predicts the drug mean for each drug-cell line combination. If the drug is not in the training set, the dataset mean is used.
        :param cell_line_id: cell line ids
        :param drug_id: drug ids
        :return: array of the same length as the input drug_id containing the drug mean
        """
        return np.array([self.predict_drug(drug) for drug in drug_id])

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

    def __init__(self, target, *args, **kwargs):
        super().__init__(target, args, kwargs)
        self.cell_line_means = None
        self.dataset_mean = None

    def build_model(self, *args, **kwargs):
        pass

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_id: np.ndarray = None,
        drug_id: np.ndarray = None,
    ) -> None:
        """
        Computes the mean per cell line. If - later on - the cell line is not in the training set, the overall mean is used.
        :param output: training dataset containing the response output
        :param cell_line_id: cell line id
        :param drug_id: drug id
        """
        self.dataset_mean = np.mean(output.response)
        self.cell_line_means = {}
        for cl in np.unique(cell_line_id):
            self.cell_line_means[cl] = np.mean(output.response[cell_line_id == cl])

    def predict(
        self, cell_line_id: np.ndarray = None, drug_id: np.ndarray = None
    ) -> np.ndarray:
        """
        Predicts the cell line mean for each drug-cell line combination. If the cell line is not in the training set, the dataset mean is used.
        :param cell_line_id: cell line ids
        :param drug_id: drug ids
        :return: array of the same length as the input cell_line_id containing the cell line mean
        """
        return np.array([self.predict_cl(cl) for cl in cell_line_id])

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
