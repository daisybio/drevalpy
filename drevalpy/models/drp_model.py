"""
Contains the DRPModel class and the SingleDrugModel class.

The DRPModel class is an abstract wrapper class for drug response prediction models. The SingleDrugModel class is an
abstract wrapper class for single drug models.
"""

import inspect
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import yaml
from numpy.typing import ArrayLike
from sklearn.model_selection import ParameterGrid

from ..datasets.dataset import DrugResponseDataset, FeatureDataset
from ..pipeline_function import pipeline_function


class DRPModel(ABC):
    """Abstract wrapper class for drug response prediction models."""

    # Used in the pipeline!
    early_stopping = False

    @abstractmethod
    @pipeline_function
    def __init__(self, *args, **kwargs) -> None:
        """
        Creates an instance of a drug response prediction model.

        :param args: optional arguments
        :param kwargs: optional keyword arguments
        """

    @classmethod
    @pipeline_function
    def get_hyperparameter_set(cls, hyperparameter_file: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Loads the hyperparameters from a yaml file.

        :param hyperparameter_file: yaml file containing the hyperparameters
        :returns: list of hyperparameter sets
        :raises ValueError: if the hyperparameters are not in the correct format
        :raises KeyError: if the model is not found in the hyperparameters file
        """
        if hyperparameter_file is None:
            hyperparameter_file = os.path.join(os.path.dirname(inspect.getfile(cls)), "hyperparameters.yaml")

        with open(hyperparameter_file, encoding="utf-8") as f:
            try:
                hpams = yaml.safe_load(f)[cls.model_name]
            except yaml.YAMLError as exc:
                raise ValueError(f"Error in hyperparameters.yaml: {exc}") from exc
            except KeyError as key_exc:
                raise KeyError(f"Model {cls.model_name} not found in hyperparameters.yaml") from key_exc

        if hpams is None:
            return [{}]
        # each param should be a list
        for hp in hpams:
            if not isinstance(hpams[hp], list):
                hpams[hp] = [hpams[hp]]
        grid = list(ParameterGrid(hpams))
        return grid

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the model name.

        :return: model name
        """

    @property
    @abstractmethod
    def cell_line_views(self) -> list[str]:
        """
        Returns the sources the model needs as input for describing the cell line.

        :return: cell line views, e.g., ["methylation", "gene_expression", "mirna_expression",
            "mutation"]
        """

    @property
    @abstractmethod
    def drug_views(self) -> list[str]:
        """
        Returns the sources the model needs as input for describing the drug.

        :return: drug views, e.g., ["descriptors", "fingerprints", "targets"]
        """

    @abstractmethod
    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Builds the model, for models that use hyperparameters.

        :param hyperparameters: hyperparameters for the model
        """

    @abstractmethod
    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: Optional[FeatureDataset],
        drug_input: Optional[FeatureDataset] = None,
        output_earlystopping: Optional[DrugResponseDataset] = None,
    ) -> None:
        """
        Trains the model.

        :param output: training data associated with the response output
        :param cell_line_input: input associated with the cell line
        :param drug_input: input associated with the drug
        :param output_earlystopping: optional early stopping dataset
        """

    @abstractmethod
    def predict(
        self,
        drug_ids: ArrayLike,
        cell_line_ids: ArrayLike,
        drug_input: Optional[FeatureDataset] = None,
        cell_line_input: Optional[FeatureDataset] = None,
    ) -> np.ndarray:
        """
        Predicts the response for the given input.

        :param drug_ids: list of drug ids
        :param cell_line_ids: list of cell line ids
        :param cell_line_input: input associated with the cell line
        :param drug_input: input associated with the drug
        :returns: predicted response
        """

    @abstractmethod
    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load the cell line features.

        :param data_path: path to the data, e.g., data/
        :param dataset_name: name of the dataset, e.g., "GDSC2"
        :returns: FeatureDataset with the cell line features
        """

    @abstractmethod
    def load_drug_features(self, data_path: str, dataset_name: str) -> Optional[FeatureDataset]:
        """
        Load the drug features.

        :param data_path: path to the data, e.g., data/
        :param dataset_name: name of the dataset, e.g., "GDSC2"
        :returns: FeatureDataset
        """

    def get_concatenated_features(
        self,
        cell_line_view: Optional[str],
        drug_view: Optional[str],
        cell_line_ids_output: ArrayLike,
        drug_ids_output: ArrayLike,
        cell_line_input: Optional[FeatureDataset],
        drug_input: Optional[FeatureDataset],
    ) -> np.ndarray:
        """
        Concatenates the features for the given cell line and drug view.

        :param cell_line_view: gene expression, methylation, etc.
        :param drug_view: ids, fingerprints, etc.
        :param cell_line_ids_output: cell line ids
        :param drug_ids_output: drug ids
        :param cell_line_input: input associated with the cell line
        :param drug_input: input associated with the drug
        :returns: X, the feature matrix needed for, e.g., sklearn models
        :raises ValueError: if no features are provided
        """
        inputs = self.get_feature_matrices(
            cell_line_ids=cell_line_ids_output,
            drug_ids=drug_ids_output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        cell_line_features = None if cell_line_view is None else inputs.get(cell_line_view)
        drug_features = None if drug_view is None else inputs.get(drug_view)

        if cell_line_features is not None and drug_features is not None:
            x = np.concatenate((cell_line_features, drug_features), axis=1)
        elif cell_line_features is not None:
            x = cell_line_features
        elif drug_features is not None:
            x = drug_features
        else:
            raise ValueError("No features provided.")
        return x

    def get_feature_matrices(
        self,
        cell_line_ids: ArrayLike,
        drug_ids: ArrayLike,
        cell_line_input: Optional[FeatureDataset],
        drug_input: Optional[FeatureDataset],
    ) -> dict[str, np.ndarray]:
        """
        Returns the feature matrices for the given cell line and drug ids by retrieving the correct views.

        :param cell_line_ids: cell line identifiers
        :param drug_ids: drug identifiers
        :param cell_line_input: cell line omics features
        :param drug_input: drug omics features
        :returns: dictionary with the feature matrices
        :raises ValueError: if the input does not contain the correct views
        """
        cell_line_feature_matrices = {}
        if cell_line_input is not None:
            for cell_line_view in self.cell_line_views:
                if cell_line_view not in cell_line_input.get_view_names():
                    raise ValueError(f"Cell line input does not contain view {cell_line_view}")
                cell_line_feature_matrices[cell_line_view] = cell_line_input.get_feature_matrix(
                    view=cell_line_view, identifiers=cell_line_ids
                )
        drug_feature_matrices = {}
        if drug_input is not None:
            for drug_view in self.drug_views:
                if drug_view not in drug_input.get_view_names():
                    raise ValueError(f"Drug input does not contain view {drug_view}")
                drug_feature_matrices[drug_view] = drug_input.get_feature_matrix(view=drug_view, identifiers=drug_ids)
        
        return {**cell_line_feature_matrices, **drug_feature_matrices}


class SingleDrugModel(DRPModel, ABC):
    """Abstract wrapper class for single drug response prediction models."""

    early_stopping = False
    drug_views = []

    def load_drug_features(self, data_path: str, dataset_name: str) -> Optional[FeatureDataset]:
        """
        Load the drug features, unnecessary for single drug models, so this function is overwritten.

        :param data_path: path to the data, e.g., data/
        :param dataset_name: name of the dataset, e.g., "GDSC2"
        :returns: nothing because it is not needed for the single drug models
        """
        return None
