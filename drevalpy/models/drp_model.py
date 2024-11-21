"""
Contains the DRPModel class.

The DRPModel class is an abstract wrapper class for drug response prediction models.


"""

import inspect
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import yaml
from sklearn.model_selection import ParameterGrid

from ..datasets.dataset import DrugResponseDataset, FeatureDataset
from ..pipeline_function import pipeline_function


class DRPModel(ABC):
    """
    Abstract wrapper class for drug response prediction models.

    The DRPModel class is an abstract wrapper class for drug response prediction models.
    It has a boolean attribute is_single_drug_model indicating whether it is a single drug model and a boolean
    attribute early_stopping indicating whether early stopping is used.
    """

    # Used in the pipeline!
    early_stopping = False
    # Then, the model is trained per drug
    is_single_drug_model = False

    @classmethod
    @abstractmethod
    @pipeline_function
    def get_model_name(cls) -> str:
        """
        Returns the name of the model.

        :return: model name
        """

    @classmethod
    @pipeline_function
    def get_hyperparameter_set(cls) -> list[dict[str, Any]]:
        """
        Loads the hyperparameters from a yaml file which is located in the same directory as the model.

        :returns: list of hyperparameter sets
        :raises ValueError: if the hyperparameters are not in the correct format
        :raises KeyError: if the model is not found in the hyperparameters file
        """
        hyperparameter_file = os.path.join(os.path.dirname(inspect.getfile(cls)), "hyperparameters.yaml")

        with open(hyperparameter_file, encoding="utf-8") as f:
            try:
                hpams = yaml.safe_load(f)[cls.get_model_name()]
            except yaml.YAMLError as exc:
                raise ValueError(f"Error in hyperparameters.yaml: {exc}") from exc
            except KeyError as key_exc:
                raise KeyError(f"Model {cls.get_model_name()} not found in hyperparameters.yaml") from key_exc

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
    def cell_line_views(self) -> list[str]:
        """
        Returns the sources the model needs as input for describing the cell line.

        :return: cell line views, e.g., ["methylation", "gene_expression", "mirna_expression",
            "mutation"]. If the model does not use cell line features, return an empty list.
        """

    @property
    @abstractmethod
    def drug_views(self) -> list[str]:
        """
        Returns the sources the model needs as input for describing the drug.

        :return: drug views, e.g., ["descriptors", "fingerprints", "targets"]. If the model does not use drug features,
            return an empty list.
        """

    @abstractmethod
    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Builds the model, for models that use hyperparameters.

        :param hyperparameters: hyperparameters for the model

        Example::

            self.model = ElasticNet(alpha=hyperparameters["alpha"], l1_ratio=hyperparameters["l1_ratio"])
        """

    @abstractmethod
    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
    ) -> None:
        """
        Trains the model.

        :param output: training data associated with the response output
        :param cell_line_input: input associated with the cell line, required for all models
        :param drug_input: input associated with the drug, optional because single drug models do not use drug features
        :param output_earlystopping: optional early stopping dataset
        """

    @abstractmethod
    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the response for the given input.

        :param drug_ids: list of drug ids, also used for single drug models, there it is just an array containing the
            same drug id
        :param cell_line_ids: list of cell line ids
        :param cell_line_input: input associated with the cell line, required for all models
        :param drug_input: input associated with the drug, optional because single drug models do not use drug features
        :returns: predicted response
        """

    @abstractmethod
    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load the cell line features before the train/predict method is called.

        Required to implement for all models. Could, e.g., call get_multiomics_feature_dataset() or
        load_and_reduce_gene_features() from models/utils.py.

        :param data_path: path to the data, e.g., data/
        :param dataset_name: name of the dataset, e.g., "GDSC2"
        :returns: FeatureDataset with the cell line features
        """

    @abstractmethod
    def load_drug_features(self, data_path: str, dataset_name: str) -> Optional[FeatureDataset]:
        """
        Load the drug features before the train/predict method is called.

        Required to implement for all models that use drug features. Could, e.g.,
        call load_drug_fingerprint_features() or load_drug_ids_from_csv() from models/utils.py.

        For single drug models, this method can return None.

        :param data_path: path to the data, e.g., data/
        :param dataset_name: name of the dataset, e.g., "GDSC2"
        :returns: FeatureDataset or None
        """

    def get_concatenated_features(
        self,
        cell_line_view: Optional[str],
        drug_view: Optional[str],
        cell_line_ids_output: np.ndarray,
        drug_ids_output: np.ndarray,
        cell_line_input: Optional[FeatureDataset],
        drug_input: Optional[FeatureDataset],
    ) -> np.ndarray:
        """
        Concatenates the features to an input matrix X for the given cell line and drug views.

        :param cell_line_view: gene expression, methylation, etc.
        :param drug_view: ids, fingerprints, etc.
        :param cell_line_ids_output: cell line ids
        :param drug_ids_output: drug ids
        :param cell_line_input: input associated with the cell line
        :param drug_input: input associated with the drug
        :returns: X, the feature matrix needed for, e.g., sklearn models
        :raises ValueError: if no features are provided

        This can, e.g., be done in the training method to produce a large input feature matrix for the model where
        the rows are the samples and the columns are the cell line and drug features concatenated. This method is an
        alternative to using DataLoaders. It is used for models operating on the whole input matrix at once.

        Example::

            x = self.get_concatenated_features(
                cell_line_view="gene_expression",
                drug_view="fingerprints",
                cell_line_ids_output=output.cell_line_ids,
                drug_ids_output=output.drug_ids,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
            )
            self.model.fit(x, output.response)
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
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
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

        This can e.g., done to produce the input for the predict() method for deep learning models:
        Example::

            input_data = self.get_feature_matrices(
                cell_line_ids=cell_line_ids,
                drug_ids=drug_ids,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
            )
            (
                gene_expression,
                mutations,
                cnvs
            ) = (
                input_data["gene_expression"],
                input_data["mutations"],
                input_data["copy_number_variation_gistic"]
            )
            return self.model.predict(gene_expression, mutations, cnvs)

        Or to produce separate inputs for the train()/predict() method for other models if the model does not operate
        on the concatenated input matrix::

            inputs = self.get_feature_matrices(
                cell_line_ids=output.cell_line_ids,
                drug_ids=output.drug_ids,
                cell_line_input=cell_line_input,
                 drug_input=drug_input,
            )
            (
                gene_expression,
                methylation,
                mutations,
                copy_number_variation_gistic,
                fingerprints,
            ) = (
                inputs["gene_expression"],
                inputs["methylation"],
                inputs["mutations"],
                inputs["copy_number_variation_gistic"],
                inputs["fingerprints"],
            )
            self.model.fit(
                gene_expression, methylation, mutations, copy_number_variation_gistic, fingerprints, output.response
            )
        """
        cell_line_feature_matrices = {}
        if cell_line_input is not None:
            for cell_line_view in self.cell_line_views:
                if cell_line_view not in cell_line_input.view_names:
                    raise ValueError(f"Cell line input does not contain view {cell_line_view}")
                cell_line_feature_matrices[cell_line_view] = cell_line_input.get_feature_matrix(
                    view=cell_line_view, identifiers=cell_line_ids
                )
        drug_feature_matrices = {}
        if drug_input is not None:
            for drug_view in self.drug_views:
                if drug_view not in drug_input.view_names:
                    raise ValueError(f"Drug input does not contain view {drug_view}")
                drug_feature_matrices[drug_view] = drug_input.get_feature_matrix(view=drug_view, identifiers=drug_ids)

        return {**cell_line_feature_matrices, **drug_feature_matrices}
