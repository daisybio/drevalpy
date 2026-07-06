"""
Implements the naive predictor models.

The naive predictor models are simple models that predict the mean of the response values. The NaivePredictor
predicts the overall mean of the response, the NaiveCellLineMeanPredictor predicts the mean of the response per cell
line, and the NaiveDrugMeanPredictor predicts the mean of the response per drug.
The NaiveTissueMeanPredictor predicts the mean of the response per tissue.
The NaiveTissueDrugMeanPredictor predicts the mean of the response per tissue-drug combination.
The NaiveMeanEffectsPredictor predicts the response as the overall mean plus tissue effect,
cell line residual effect, and drug effect and should be the strongest naive baseline.

"""

import numpy as np

from drevalpy.components.register_builtins import ensure_components_registered
from drevalpy.data.features import (
    load_cl_ids_and_tissues_from_csv,
    load_cl_ids_from_csv,
    load_drug_ids_from_csv,
    load_tissues_from_csv,
)
from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, TISSUE_IDENTIFIER
from drevalpy.models._component_bridge import (
    ComponentDRPBridge,
    save_component_stack,
)
from drevalpy.models._legacy_checkpoint_loaders import (
    has_component_stack,
    load_legacy_naive_checkpoint,
    load_native_checkpoint,
)
from drevalpy.models._legacy_state_accessors import naive_state_from_bridge
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.factory import NAIVE_PREDICTOR_BY_MODEL_NAME, model_config_for_name


class NaiveModel(DRPModel):
    """
    Base class for all naive predictor models which are based on simple dataset stats.

    This class provides a shared interface and save/load mechanism for simple statistical models that
    predict drug response based on dataset means, stratified by drug, cell line, or tissue.
    """

    def __init__(self):
        """Initializes the NaiveModel base class."""
        super().__init__()
        self._component_bridge = ComponentDRPBridge()

    def _naive_state(self) -> dict:
        return naive_state_from_bridge(self._component_bridge)

    @property
    def dataset_mean(self):
        return self._naive_state().get("dataset_mean")

    @property
    def drug_means(self):
        return self._naive_state().get("drug_means")

    @property
    def cell_line_means(self):
        return self._naive_state().get("cell_line_means")

    @property
    def tissue_means(self):
        return self._naive_state().get("tissue_means")

    @property
    def tissue_drug_means(self):
        return self._naive_state().get("tissue_drug_means")

    @property
    def cell_line_effects(self):
        return self._naive_state().get("cell_line_effects")

    @property
    def drug_effects(self):
        return self._naive_state().get("drug_effects")

    @property
    def tissue_effects(self):
        return self._naive_state().get("tissue_effects", {})

    def _predictor_type(self) -> str:
        predictor_type = NAIVE_PREDICTOR_BY_MODEL_NAME.get(self.get_model_name())
        if predictor_type is None:
            msg = f"No component predictor registered for {self.get_model_name()!r}"
            raise ValueError(msg)
        return predictor_type

    def build_model(self, hyperparameters: dict):
        """
        Builds the model.

        Naive model do not require any hyperparameter tuning.

        :param hyperparameters: Dictionary of hyperparameters (not used).
        """
        ensure_components_registered()
        config = model_config_for_name(self.get_model_name())
        self._component_bridge.set_composed_config(config)

    def _ensure_built(self) -> None:
        if self._component_bridge.composed is None:
            self.build_model({})

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """Train the naive baseline via the component stack."""
        self._ensure_built()
        self._component_bridge.train(output, cell_line_input, drug_input)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """Predict via the component stack."""
        self._ensure_built()
        return self._component_bridge.predict(
            cell_line_ids,
            drug_ids,
            cell_line_input,
            drug_input,
        )

    def save(self, directory: str) -> None:
        """
        Saves the model parameters to the given directory.

        Persists the composed component stack when trained.

        :param directory: Path to the directory where the model will be saved.
        """
        self._ensure_built()
        if not self._component_bridge.is_trained():
            msg = "Cannot save: component stack is not trained"
            raise RuntimeError(msg)
        save_component_stack(self._component_bridge, directory, hyperparameters={})

    @classmethod
    def load(cls, directory: str) -> "NaiveModel":
        """
        Loads the model parameters from the given directory.

        :param directory: Path to the directory where the model is saved.
        :return: An instance of NaiveModel with the loaded parameters.
        """
        if has_component_stack(directory):
            instance = cls()
            instance.build_model({})
            load_native_checkpoint(instance, directory)
            return instance

        instance = cls()
        load_legacy_naive_checkpoint(instance, directory)
        return instance


class NaivePredictor(NaiveModel):
    """Naive predictor model that predicts the overall mean of the response."""

    cell_line_views = [CELL_LINE_IDENTIFIER]
    drug_views = [DRUG_IDENTIFIER]

    def __init__(self):
        """
        Initializes the model.

        Sets the dataset mean to None, which is initialized in the train method.
        """
        super().__init__()

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: NaivePredictor
        """
        return "NaivePredictor"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features, in this case the cell line ids.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the cell line ids
        """
        return load_cl_ids_from_csv(data_path, dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the drug features, in this case the drug ids.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the drug ids
        """
        return load_drug_ids_from_csv(data_path, dataset_name)


class NaiveDrugMeanPredictor(NaiveModel):
    """Naive predictor model that predicts the mean of the response per drug."""

    cell_line_views = [CELL_LINE_IDENTIFIER]
    drug_views = [DRUG_IDENTIFIER]

    def __init__(self):
        """
        Initializes the model.

        Drug means and dataset mean are set to None, which are initialized in the train method.
        """
        super().__init__()

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: NaiveDrugMeanPredictor
        """
        return "NaiveDrugMeanPredictor"

    def predict_drug(self, drug_id: str):
        """
        Predicts the mean of the response for a given drug.

        If the drug is not in the training set, the dataset mean is used.

        :param drug_id: ID of the drug
        :return: predicted response
        """
        if drug_id in self.drug_means:
            return self.drug_means[drug_id]
        return self.dataset_mean

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: Path to the data.
        :param dataset_name: Name of the dataset.
        :return: FeatureDataset containing the cell line IDs.
        """
        return load_cl_ids_from_csv(data_path, dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the drug features, in this case the drug ids.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the drug ids
        """
        return load_drug_ids_from_csv(data_path, dataset_name)


class NaiveCellLineMeanPredictor(NaiveModel):
    """Naive predictor model that predicts the mean of the response per cell line."""

    cell_line_views = [CELL_LINE_IDENTIFIER]
    drug_views = [DRUG_IDENTIFIER]

    def __init__(self):
        """
        Initializes the model.

        Cell line means and dataset mean are set to None, which are initialized in the train method.
        """
        super().__init__()

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: NaiveCellLineMeanPredictor
        """
        return "NaiveCellLineMeanPredictor"

    def predict_cl(self, cl_id: str) -> float:
        """
        Predicts the mean of the response for a given cell line.

        If the cell line is not in the training set, the dataset mean is used.

        :param cl_id: Cell line ID
        :return: predicted response
        """
        if cl_id in self.cell_line_means:
            return self.cell_line_means[cl_id]
        return self.dataset_mean

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features, in this case the cell line ids.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the cell line ids
        """
        return load_cl_ids_from_csv(data_path, dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the drug features.

        :param data_path: Path to the data.
        :param dataset_name: Name of the dataset.
        :return: FeatureDataset containing the drug IDs.
        """
        return load_drug_ids_from_csv(data_path, dataset_name)


class NaiveTissueMeanPredictor(NaiveModel):
    """Naive predictor model that predicts the mean of the response per tissue."""

    cell_line_views = [TISSUE_IDENTIFIER]
    drug_views = []

    def __init__(self):
        """
        Initializes the model.

        Tissue means and dataset mean are set to None, which are initialized in the train method.
        """
        super().__init__()

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: NaiveTissueMeanPredictor
        """
        return "NaiveTissueMeanPredictor"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features, in this case the tissue annotations.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the tissue ids
        """
        return load_tissues_from_csv(data_path, dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the drug features.

        :param data_path: Path to the data.
        :param dataset_name: Name of the dataset.
        :return: FeatureDataset containing the drug IDs.
        """
        return load_drug_ids_from_csv(data_path, dataset_name)


class NaiveMeanEffectsPredictor(NaiveModel):
    """
    ANOVA-like predictor model.

    Predicts the response as:
    response = overall_mean + tissue_effect + cell_line_residual_effect + drug_effect.

    Here:
        - tissue_effect = (tissue mean - overall_mean)
        - cell_line_residual_effect = (cell line mean - tissue mean for that cell line)
        - drug_effect = (drug mean - overall_mean)

    This formulation avoids double-counting tissue signal already captured by cell line means.
    For unseen cell lines with a known tissue, the tissue effect provides a fallback.
    If tissue information is not available, this model falls back to the previous formulation:
    response = overall_mean + cell_line_effect + drug_effect.
    """

    cell_line_views = [CELL_LINE_IDENTIFIER]
    drug_views = [DRUG_IDENTIFIER]

    def __init__(self):
        """
        Initializes the NaiveMeanEffectsPredictor model.

        The overall dataset mean, tissue effects, cell line residual effects, and drug effects
        are initialized to None and empty dictionaries, respectively.
        """
        super().__init__()

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the name of the model.

        :return: The name of the model as a string.
        """
        return "NaiveMeanEffectsPredictor"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: Path to the data.
        :param dataset_name: Name of the dataset.
        :return: FeatureDataset containing the cell line IDs and tissue annotations, if available.
        """
        return load_cl_ids_and_tissues_from_csv(data_path, dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the drug features.

        :param data_path: Path to the data.
        :param dataset_name: Name of the dataset.
        :return: FeatureDataset containing the drug IDs.
        """
        return load_drug_ids_from_csv(data_path, dataset_name)


class NaiveTissueDrugMeanPredictor(NaiveModel):
    """
    Naive predictor model that predicts the mean of the response per tissue-drug combination.

    This model combines tissue and drug information to predict the mean response aggregated across
    all cell lines from the same tissue tested on the same drug. If a (tissue, drug) combination
    was not seen during training, it falls back to the overall dataset mean.
    """

    cell_line_views = [TISSUE_IDENTIFIER]
    drug_views = [DRUG_IDENTIFIER]

    def __init__(self):
        """
        Initializes the model.

        Tissue-drug means and dataset mean are set to None, which are initialized in the train method.
        """
        super().__init__()

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: NaiveTissueDrugMeanPredictor
        """
        return "NaiveTissueDrugMeanPredictor"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features, in this case the tissue annotations.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the tissue ids
        """
        return load_tissues_from_csv(data_path, dataset_name)

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the drug features, in this case the drug ids.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the drug ids
        """
        return load_drug_ids_from_csv(data_path, dataset_name)
