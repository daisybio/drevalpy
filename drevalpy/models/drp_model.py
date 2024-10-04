"""
Contains the DRPModel class, which is an abstract wrapper class for drug response prediction
models, the SingleDrugModel class, which is an abstract wrapper class for single drug models and
CompositeDrugModel class, which transforms multiple separate single drug response prediction models
into a global model by applying a separate model for each drug.
"""

from abc import ABC, abstractmethod
import inspect
import os
from typing import Any, Dict, Optional, Type, List
import warnings
import numpy as np
from numpy.typing import ArrayLike
import yaml
from sklearn.model_selection import ParameterGrid

from ..datasets.dataset import DrugResponseDataset, FeatureDataset


class DRPModel(ABC):
    """
    Abstract wrapper class for drug response prediction models.
    """

    early_stopping = False

    def __init__(self, *args, **kwargs):
        """
        Creates an instance of a drug response prediction model.
        :param model_name: model name for displaying results
        :param args: optional arguments
        :param kwargs: optional keyword arguments
        """

    @classmethod
    def get_hyperparameter_set(cls, hyperparameter_file: Optional[str] = None):
        """
        Loads the hyperparameters from a yaml file.
        :param hyperparameter_file: yaml file containing the hyperparameters
        :return:
        """
        if hyperparameter_file is None:
            hyperparameter_file = os.path.join(
                os.path.dirname(inspect.getfile(cls)), "hyperparameters.yaml"
            )

        with open(hyperparameter_file, "r", encoding="utf-8") as f:
            hpams = yaml.load(f, Loader=yaml.FullLoader)[cls.model_name]
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
    def model_name(self):
        """
        Returns the model name.
        :return: model name
        """

    @property
    @abstractmethod
    def cell_line_views(self):
        """
        Returns the sources the model needs as input for describing the cell line.
        :return: cell line views, e.g., ["methylation", "gene_expression", "mirna_expression",
        "mutation"]
        """

    @property
    @abstractmethod
    def drug_views(self):
        """
        Returns the sources the model needs as input for describing the drug.
        :return: drug views, e.g., ["descriptors", "fingerprints", "targets"]
        """

    @abstractmethod
    def build_model(self, hyperparameters: Dict[str, Any]):
        """
        Builds the model, for models that use hyperparameters.
        """

    @abstractmethod
    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
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
        drug_input: FeatureDataset = None,
        cell_line_input: FeatureDataset = None,
    ) -> np.ndarray:
        """
        Predicts the response for the given input.

        """

    def save(self, path):
        """
        Saves the model.
        :param path: path to save the model
        """

    def load(self, path):
        """
        Loads the model.
        :param path: path to load the model
        """

    @abstractmethod
    def load_cell_line_features(
        self, data_path: str, dataset_name: str
    ) -> FeatureDataset:
        """
        :return: FeatureDataset
        """

    @abstractmethod
    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        :return: FeatureDataset
        """

    def get_concatenated_features(
        self,
        cell_line_view: str,
        drug_view: Optional[str],
        cell_line_ids_output: ArrayLike,
        drug_ids_output: ArrayLike,
        cell_line_input: Optional[FeatureDataset],
        drug_input: Optional[FeatureDataset],
    ):
        """
        Concatenates the features for the given cell line and drug view.
        :param cell_line_view:
        :param drug_view:
        :param cell_line_ids_output:
        :param drug_ids_output:
        :param cell_line_input:
        :param drug_input:
        :return: X, the feature matrix needed for, e.g., sklearn models
        """
        inputs = self.get_feature_matrices(
            cell_line_ids=cell_line_ids_output,
            drug_ids=drug_ids_output,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        cell_line_features = inputs.get(cell_line_view)
        drug_features = inputs.get(drug_view)

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
    ):
        """
        Returns the feature matrices for the given cell line and drug ids by retrieving the
        correct views.
        :param cell_line_ids:
        :param drug_ids:
        :param cell_line_input:
        :param drug_input:
        :return:
        """
        cell_line_feature_matrices = {}
        if cell_line_input is not None:
            for cell_line_view in self.cell_line_views:
                if cell_line_view not in cell_line_input.get_view_names():
                    raise ValueError(
                        f"Cell line input does not contain view {cell_line_view}"
                    )
                cell_line_feature_matrices[cell_line_view] = (
                    cell_line_input.get_feature_matrix(
                        view=cell_line_view, identifiers=cell_line_ids
                    )
                )
        drug_feature_matrices = {}
        if drug_input is not None:
            for drug_view in self.drug_views:
                if drug_view not in drug_input.get_view_names():
                    raise ValueError(f"Drug input does not contain view {drug_view}")
                drug_feature_matrices[drug_view] = drug_input.get_feature_matrix(
                    view=drug_view, identifiers=drug_ids
                )

        return {**cell_line_feature_matrices, **drug_feature_matrices}


class SingleDrugModel(DRPModel, ABC):
    """
    Abstract wrapper class for single drug response prediction models.
    """

    early_stopping = False
    drug_views = []

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        return None


class CompositeDrugModel(DRPModel):
    """
    Transforms multiple separate single drug response prediction models into a global model by
    applying a seperate model for each drug.
    """

    cell_line_views = None
    drug_views = []
    model_name = "CompositeDrugModel"

    def __init__(self, base_model: Type[DRPModel], *args, **kwargs):
        """
        Creates an instance of a single drug response prediction model.
        :param model_name: model name for displaying results
        """
        super().__init__(*args, **kwargs)
        self.models = {}
        self.base_model = base_model
        self.cell_line_views = base_model.cell_line_views
        self.model_name = base_model.model_name
        self.early_stopping = base_model.early_stopping

    def build_model(self, hyperparameters: Dict[str, Any]):
        """
        Builds the model.
        """
        for drug in hyperparameters:
            self.models[drug] = self.base_model()
            self.models[drug].drug_views = self.drug_views
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
        cell_line_input: FeatureDataset,
        drug_input=None,
        output_earlystopping: Optional[DrugResponseDataset] = None,
    ) -> None:
        """
        Trains the model.
        :param output: Training data associated with the response output
        :param cell_line_input: Input associated with the cell line
        :param drug_input: Not needed for the single drug models
        :param output_earlystopping: Optional. Training data associated with the early stopping
        output
        """
        drugs = np.unique(output.drug_ids)
        for i, drug in enumerate(drugs):
            assert drug in self.models, (
                f"Drug {drug} not in models. Maybe the CompositeDrugModel was not built or drug "
                f"missing from train data."
            )
            print(f"Training model for drug {drug} ({i+1}/{len(drugs)})")
            output_mask = output.drug_ids == drug
            output_drug = output.copy()
            output_drug.mask(output_mask)
            output_earlystopping_drug = None
            if output_earlystopping is not None:
                output_earlystopping_mask = output_earlystopping.drug_ids == drug
                output_earlystopping_drug = output_earlystopping.copy()
                output_earlystopping_drug.mask(output_earlystopping_mask)

            self.models[drug].train(
                output=output_drug,
                cell_line_input=cell_line_input,
                output_earlystopping=output_earlystopping_drug,
            )

    def predict(
        self,
        drug_ids: List[str],
        cell_line_ids: List[str],
        drug_input=None,
        cell_line_input: FeatureDataset = None,
    ) -> np.ndarray:
        """
        Predicts the response for the given input.
        :param drug_ids: list of drug ids
        :param cell_line_ids: list of cell line ids
        :param cell_line_input: input associated with the cell line
        :param drug_input: not needed for the single drug models
        :return: predicted response
        """
        prediction = np.zeros_like(drug_ids, dtype=float)
        for drug in np.unique(drug_ids):
            mask = drug_ids == drug
            if drug not in self.models:
                prediction[mask] = np.nan
            else:
                prediction[mask] = self.models[drug].predict(
                    drug_ids=drug,
                    cell_line_ids=cell_line_ids[mask],
                    cell_line_input=cell_line_input,
                )
        if np.any(np.isnan(prediction)):
            warnings.warn(
                "SingleDRPModel Warning: Some drugs were not in the training set. Prediction is "
                "NaN. Maybe a SingleDRPModel was used in an LDO setting."
            )
        return prediction
