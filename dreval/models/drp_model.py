from abc import ABC, abstractmethod
import inspect
import os
from typing import Dict, List, Optional

import yaml
from ..datasets.dataset import DrugResponseDataset, FeatureDataset
import numpy as np
from sklearn.model_selection import ParameterGrid


class DRPModel(ABC):
    """
    Abstract wrapper class for drug response prediction models.
    """
    early_stopping = False

    def __init__(self, target, *args, **kwargs):
        """
        Creates an instance of a drug response prediction model.
        :param model_name: model name for displaying results
        :param target: target value, e.g., IC50, EC50, AUC, classification
        :param args: optional arguments
        :param kwargs: optional keyword arguments
        """
        self.target = target

    @classmethod
    def get_hyperparameter_set(cls, hyperparameter_file: Optional[str] = None):
        # load yaml file with hyperparameters
        if hyperparameter_file is None:
            hyperparameter_file = os.path.join(os.path.dirname(inspect.getfile(cls)), "hyperparameters.yaml")

        hpams = yaml.load(open(hyperparameter_file), Loader=yaml.FullLoader)[cls.model_name]
        if hpams is None:
            return [{}]

        grid = list(ParameterGrid(hpams))
        return grid

    @property
    @abstractmethod
    def model_name(self):
        """
        Returns the model name.
        :return: model name
        """
        pass

    @property
    @abstractmethod
    def cell_line_views(self):
        """
        Returns the sources the model needs as input for describing the cell line.
        :return: cell line views, e.g., ["methylation", "gene_expression", "mirna_expression", "mutation"]
        """
        pass

    @property
    @abstractmethod
    def drug_views(self):
        """
        Returns the sources the model needs as input for describing the drug.
        :return: drug views, e.g., ["descriptors", "fingerprints", "targets"]
        """
        pass

    @abstractmethod
    def build_model(self, *args, **kwargs):
        """
        Builds the model.
        """
        pass

    @abstractmethod
    def train(
            self,
            output: DrugResponseDataset,
            output_earlystopping: Optional[DrugResponseDataset] = None,
            **inputs: Dict[str, np.ndarray]
    ) -> None:
        """
        Trains the model. Call the respective function from models_code here.
        :param output: training data associated with the response output
        :param output_earlystopping: optional early stopping dataset
        :param inputs: input data Dict of numpy arrays
        """
        pass

    @abstractmethod
    def predict(self, **inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predicts the response for the given input. 

        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Saves the model to models_saved.
        :param path: path to save the model
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Loads the model.
        :param path: path to load the model
        """
        pass

    @abstractmethod
    def load_cell_line_features(self, data_path: str, dataset_name:str) -> FeatureDataset:
        """
        :return: FeatureDataset
        """
        pass

    @abstractmethod
    def load_drug_features(self, data_path: str, dataset_name:str) -> FeatureDataset:
        """
        :return: FeatureDataset
        """
        pass

    def get_feature_matrices(
            self,
            cell_line_ids: np.ndarray,
            drug_ids: np.ndarray,
            cell_line_input: FeatureDataset,
            drug_input: FeatureDataset,
    ):
        cell_line_feature_matrices = {}
        for cell_line_view in self.cell_line_views:
            if cell_line_view not in cell_line_input.get_view_names():
                raise ValueError(f"Cell line input does not contain view {cell_line_view}")
            cell_line_feature_matrices[cell_line_view] = cell_line_input.get_feature_matrix(cell_line_view,
                                                                                            cell_line_ids)
        drug_feature_matrices = {}
        for drug_view in self.drug_views:
            if drug_view not in drug_input.get_view_names():
                raise ValueError(f"Drug input does not contain view {drug_view}")
            drug_feature_matrices[drug_view] = drug_input.get_feature_matrix(drug_view, drug_ids)

        return {**cell_line_feature_matrices, **drug_feature_matrices}


class SingleDRPModel(DRPModel, ABC):
    """
    Abstract wrapper class for single drug response prediction models.
    """

    def __init__(self, model_name, target, config_path, *args, **kwargs):
        """
        Creates an instance of a single drug response prediction model.
        :param model_name: model name for displaying results
        :param target: target value, e.g., IC50, EC50, AUC, classification
        :param args: optional arguments
        :param kwargs: optional keyword arguments
        """
        super(SingleDRPModel, self).__init__(model_name, target, config_path, *args, **kwargs)

    @property
    @abstractmethod
    def cell_line_views(self):
        """
        Returns the sources the model needs as input for describing the cell line.
        :return: cell line views, e.g., ["methylation", "gene_expression", "mirna_expression", "mutation"]
        """
        pass

    @property
    @abstractmethod
    def drug_views(self):
        """
        Returns the sources the model needs as input for describing the drug.
        :return: drug views, e.g., ["descriptors", "fingerprints", "targets"]
        """
        pass

    @abstractmethod
    def build_model(self, *args, **kwargs):
        """
        Builds the model.
        """
        pass

    def train(
            self,
            cell_line_input: FeatureDataset,
            drug_input: str,
            output: DrugResponseDataset,
    ):
        """
        Trains the model.
        :param cell_line_input: training data associated with the cell line input
        :param drug_input: drug name
        :param output: training data associated with the response output
        """
        self.train_drug(cell_line_input, drug_input, output)

    @abstractmethod
    def train_drug(
            self,
            cell_line_input: FeatureDataset,
            drug_name: str,
            output: DrugResponseDataset,
    ):
        """
        Trains one model per drug.
        :param cell_line_input: training data associated with the cell line input
        :param drug_name: drug name
        :param output: training data associated with the response output
        """
        pass

    def predict(self, cell_line_input: FeatureDataset, drug_input: str) -> np.ndarray:
        """
        Predicts the response for the given input.
        :param cell_line_input: input associated with the cell line
        :param drug_input: drug name
        :return: predicted response
        """
        self.predict_drug(cell_line_input, drug_input)

    @abstractmethod
    def predict_drug(self, cell_line_input: FeatureDataset, drug_name: str):
        """
        Predicts the response for the given single drug.
        :param cell_line_input: input associated with the cell line
        :param drug_name: drug name
        :return: predicted response
        """
        raise NotImplementedError("predict_drug method not implemented")

    @abstractmethod
    def save(self, path):
        """
        Saves the model to models_saved.
        :param path: path to save the model
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Loads the model.
        :param path: path to load the model
        """
        pass
