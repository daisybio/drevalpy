from abc import ABC, abstractmethod
import inspect
import os
from typing import Any, Dict, Optional, Type
import warnings

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
            hyperparameter_file = os.path.join(
                os.path.dirname(inspect.getfile(cls)), "hyperparameters.yaml"
            )

        hpams = yaml.load(open(hyperparameter_file), Loader=yaml.FullLoader)[
            cls.model_name
        ]
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

    # TODO maybe this does not need to be abstract since some models do not require hpams
    @abstractmethod
    def build_model(self, hyperparameters: Dict[str, Any], *args, **kwargs):
        """
        Builds the model, for models that use hyperparameters.
        """
        pass

    @abstractmethod
    def train(
        self,
        output: DrugResponseDataset,
        output_earlystopping: Optional[DrugResponseDataset] = None,
        **inputs: Dict[str, np.ndarray],
    ) -> None:
        """
        Trains the model.
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

    def save(self, path):
        """
        Saves the model.
        :param path: path to save the model
        """
        pass

    def load(self, path):
        """
        Loads the model.
        :param path: path to load the model
        """
        pass

    @abstractmethod
    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
        """
        :return: FeatureDataset
        """
        pass

    @abstractmethod
    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
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
                raise ValueError(
                    f"Cell line input does not contain view {cell_line_view}"
                )
            cell_line_feature_matrices[cell_line_view] = (
                cell_line_input.get_feature_matrix(cell_line_view, cell_line_ids)
            )
        drug_feature_matrices = {}
        for drug_view in self.drug_views:
            if drug_view not in drug_input.get_view_names():
                raise ValueError(f"Drug input does not contain view {drug_view}")
            drug_feature_matrices[drug_view] = drug_input.get_feature_matrix(
                drug_view, drug_ids
            )

        return {**cell_line_feature_matrices, **drug_feature_matrices}


class SingleDrugModel(DRPModel, ABC):
    """
    Abstract wrapper class for single drug response prediction models.
    """

    early_stopping = False
    drug_views = []

    def __init__(self, target: str, *args, **kwargs):
        """
        Creates an instance of a single drug response prediction model.
        :param model_name: model name for displaying results
        :param target: target value, e.g., IC50, EC50, AUC, classification
        """
        super().__init__(target=target)
        self.target = target

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        return None


class CompositeDrugModel(DRPModel):
    """
    Transforms multiple separate single drug response prediction models into a global model by applying a seperate model for each drug.
    """

    cell_line_views = None
    drug_views = []
    model_name = "CompositeDrugModel"

    def __init__(self, target: str, base_model: Type[DRPModel], *args, **kwargs):
        """
        Creates an instance of a single drug response prediction model.
        :param model_name: model name for displaying results
        :param target: target value, e.g., IC50, EC50, AUC, classification
        """
        super().__init__(target=target, *args, **kwargs)
        self.models = {}
        self.base_model = base_model
        self.cell_line_views = base_model.cell_line_views
        self.model_name = base_model.model_name
        self.early_stopping = base_model.early_stopping

    def build_model(self, hyperparameters: Dict[str, Any], *args, **kwargs):
        """
        Builds the model.
        """
        for drug in hyperparameters:
            self.models[drug] = self.base_model(target=self.target)
            self.models[drug].build_model(hyperparameters[drug])

    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
        return list(self.models.values())[0].load_cell_line_features(
            data_path=data_path, dataset_name=dataset_name
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        return None

    def train(
        self,
        output: DrugResponseDataset,
        output_earlystopping: Optional[DrugResponseDataset] = None,
        **inputs: Dict[str, np.ndarray],
    ) -> None:
        """
        Trains the model.
        :param output: Training data associated with the response output
        :param output_earlystopping: Optional. Training data associated with the early stopping output
        :param inputs: Dictionary containing input data associated with different views
        """
        drugs = np.unique(output.drug_ids)
        for i, drug in enumerate(drugs):
            assert (
                drug in self.models
            ), f"Drug {drug} not in models. Maybe the CompositeDrugModel was not built or drug missing from train data."
            print(f"Training model for drug {drug} ({i+1}/{len(drugs)})")
            output_mask = output.drug_ids == drug
            output_drug = output.copy()
            output_drug.mask(output_mask)

            inputs_drug = {
                view: data[output_mask]
                for view, data in inputs.items()
                if not view.endswith("_earlystopping")
            }

            output_earlystopping_drug = None
            if output_earlystopping is not None:
                output_earlystopping_mask = output_earlystopping.drug_ids == drug
                output_earlystopping_drug = output_earlystopping.copy()
                output_earlystopping_drug.mask(output_earlystopping_mask)
                inputs_drug.update(
                    {
                        view: data[output_earlystopping_mask]
                        for view, data in inputs.items()
                        if view.endswith("_earlystopping")
                    }
                )

                self.models[drug].train(
                    output=output_drug,
                    output_earlystopping=output_earlystopping_drug,
                    **inputs_drug,
                )
            else:
                assert self.models[drug] is not None, f"none for drug {drug}"
                self.models[drug].train(output=output_drug, **inputs_drug)

    def predict(self, drug_ids, **inputs) -> np.ndarray:
        """
        Predicts the response for the given input.
        :param cell_line_input: input associated with the cell line
        :param drug_input: drug name
        :return: predicted response
        """
        prediction = np.zeros_like(drug_ids, dtype=float)
        for drug in np.unique(drug_ids):
            mask = drug_ids == drug
            inputs_drug = {view: data[mask] for view, data in inputs.items()}

            if drug not in self.models:
                prediction[mask] = np.nan
            else:
                prediction[mask] = self.models[drug].predict(**inputs_drug)
        if np.any(np.isnan(prediction)):
            warnings.warn(
                "SingleDRPModel Warning: Some drugs were not in the training set. Prediction is NaN Maybe a SingleDRPModel was used in an LDO setting."
            )
        return prediction
