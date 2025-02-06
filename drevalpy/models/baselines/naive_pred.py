"""
Implements the naive predictor models.

The naive predictor models are simple models that predict the mean of the response values. The NaivePredictor
predicts the overall mean of the response, the NaiveCellLineMeanPredictor predicts the mean of the response per cell
line, and the NaiveDrugMeanPredictor predicts the mean of the response per drug.
The NaiveMeanEffectsPredictor predicts the response as the overall mean plus the cell line effect
plus the drug effect and should be the strongest naive baseline.

"""

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.utils import load_cl_ids_from_csv, load_drug_ids_from_csv, unique


class NaivePredictor(DRPModel):
    """Naive predictor model that predicts the overall mean of the response."""

    cell_line_views = ["cell_line_id"]
    drug_views = ["drug_id"]

    def __init__(self):
        """
        Initializes the model.

        Sets the dataset mean to None, which is initialized in the train method.
        """
        super().__init__()
        self.dataset_mean = None

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: NaivePredictor
        """
        return "NaivePredictor"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters. Not needed for the NaivePredictor.

        :param hyperparameters: Hyperparameters for the model, not needed
        """
        pass

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Computes the overall mean of the output response values and saves them.

        :param output: training dataset containing the response output
        :param cell_line_input: not needed
        :param drug_input: not needed
        :param output_earlystopping: not needed
        :param model_checkpoint_dir: not needed
        """
        self.dataset_mean = np.mean(output.response)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the dataset mean for each drug-cell line combination.

        :param cell_line_ids: cell line ids
        :param drug_ids: not needed
        :param cell_line_input: not needed
        :param drug_input: not needed
        :return: array of the same length as the input cell line id containing the dataset mean
        """
        return np.full(cell_line_ids.shape[0], self.dataset_mean)

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


class NaiveDrugMeanPredictor(DRPModel):
    """Naive predictor model that predicts the mean of the response per drug."""

    cell_line_views = ["cell_line_id"]
    drug_views = ["drug_id"]

    def __init__(self):
        """
        Initializes the model.

        Drug means and dataset mean are set to None, which are initialized in the train method.
        """
        super().__init__()
        self.drug_means = None
        self.dataset_mean = None

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: NaiveDrugMeanPredictor
        """
        return "NaiveDrugMeanPredictor"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters. Not needed for the NaiveDrugMeanPredictor.

        :param hyperparameters: Hyperparameters for the model, not needed
        """
        pass

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "None",
    ) -> None:
        """
        Computes the mean per drug. If - later on - the drug is not in the training set, the overall mean is used.

        :param output: training dataset containing the response output
        :param cell_line_input: not needed
        :param drug_input: drug id
        :param output_earlystopping: not needed
        :param model_checkpoint_dir: not needed
        :raises ValueError: If drug_input is None
        """
        if drug_input is None:
            raise ValueError("drug_input (drug_id) is required for the NaiveDrugMeanPredictor.")
        drug_ids = drug_input.get_feature_matrix(view="drug_id", identifiers=output.drug_ids)
        self.dataset_mean = np.mean(output.response)
        self.drug_means = {}

        for drug_response, drug_feature in zip(unique(output.drug_ids), unique(drug_ids), strict=True):
            responses_drug = output.response[drug_feature == output.drug_ids]
            if len(responses_drug) > 0:
                # prevent nan response
                self.drug_means[drug_response] = np.mean(responses_drug)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the drug mean for each drug-cell line combination.

        If the drug is not in the training set, the dataset mean is used.

        :param cell_line_ids: not needed
        :param drug_ids: drug ids
        :param cell_line_input: not needed
        :param drug_input: not needed
        :return: array of the same length as the input drug_id containing the drug mean
        """
        return np.array([self.predict_drug(drug) for drug in drug_ids])

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


class NaiveCellLineMeanPredictor(DRPModel):
    """Naive predictor model that predicts the mean of the response per cell line."""

    cell_line_views = ["cell_line_id"]
    drug_views = ["drug_id"]

    def __init__(self):
        """
        Initializes the model.

        Cell line means and dataset mean are set to None, which are initialized in the train method.
        """
        super().__init__()
        self.cell_line_means = None
        self.dataset_mean = None

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: NaiveCellLineMeanPredictor
        """
        return "NaiveCellLineMeanPredictor"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model from hyperparameters. Not needed for the NaiveCellLineMeanPredictor.

        :param hyperparameters: not needed
        """
        pass

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "None",
    ) -> None:
        """
        Computes the mean per cell line.

        If - later on - the cell line is not in the training set, the overall mean is used.
        :param output: training dataset containing the response output
        :param cell_line_input: cell line inputs
        :param drug_input: not needed
        :param output_earlystopping: not needed
        :param model_checkpoint_dir: not needed
        """
        cell_line_ids = cell_line_input.get_feature_matrix(view="cell_line_id", identifiers=output.cell_line_ids)
        self.dataset_mean = np.mean(output.response)
        self.cell_line_means = {}

        for cell_line_response, cell_line_feature in zip(
            unique(output.cell_line_ids), unique(cell_line_ids), strict=True
        ):
            responses_cl = output.response[cell_line_feature == output.cell_line_ids]
            if len(responses_cl) > 0:
                # prevent nan response
                self.cell_line_means[cell_line_response] = np.mean(responses_cl)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the cell line mean for each drug-cell line combination.

        If the cell line is not in the training set, the dataset mean is used.

        :param cell_line_ids: cell line ids
        :param drug_ids: not needed
        :param cell_line_input: not needed
        :param drug_input: not needed
        :return: array of the same length as the input cell_line_id containing the cell line mean
        """
        return np.array([self.predict_cl(cl) for cl in cell_line_ids])

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
        Loads the drug features, in this case the drug ids.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the drug ids
        """
        return load_drug_ids_from_csv(data_path, dataset_name)


class NaiveMeanEffectsPredictor(DRPModel):
    """
    ANOVA-like predictor model.

    Predicts the response as:
    response = overall_mean + cell_line_effect + drug_effect.

    Here:
        - cell_line_effect = (cell line mean - overall_mean)
        - drug_effect = (drug mean - overall_mean)

    This formulation ensures that the overall mean is not counted twice.
    """

    cell_line_views = ["cell_line_id"]
    drug_views = ["drug_id"]

    def __init__(self):
        """
        Initializes the NaiveMeanEffectsPredictor model.

        The overall dataset mean, cell line effects, and drug effects are initialized to None
        and empty dictionaries, respectively.
        """
        super().__init__()
        self.dataset_mean = None
        self.cell_line_effects = {}
        self.drug_effects = {}

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the name of the model.

        :return: The name of the model as a string.
        """
        return "NaiveMeanEffectsPredictor"

    def build_model(self, hyperparameters: dict):
        """
        Builds the model.

        This model does not require any hyperparameter tuning.

        :param hyperparameters: Dictionary of hyperparameters (not used).
        """
        pass

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Trains with overall mean, cell line effects, and drug effects.

        :param output: Training dataset containing the response output.
        :param cell_line_input: Feature dataset containing cell line IDs.
        :param drug_input: Feature dataset containing drug IDs. Must not be None.
        :param output_earlystopping: Not used.
        :param model_checkpoint_dir: Not used.
        :raises ValueError: If drug_input is None.
        """
        if drug_input is None:
            raise ValueError("drug_input (drug_id) is required for ANOVAPredictor.")

        # Compute the overall mean response.
        self.dataset_mean = np.mean(output.response)

        # Obtain cell line features.
        cell_line_ids = cell_line_input.get_feature_matrix(view="cell_line_id", identifiers=output.cell_line_ids)
        cell_line_means = {}
        for cl_output, cl_feature in zip(unique(output.cell_line_ids), unique(cell_line_ids), strict=True):
            responses_cl = output.response[cl_feature == output.cell_line_ids]
            if len(responses_cl) > 0:
                cell_line_means[cl_output] = np.mean(responses_cl)

        # Obtain drug features.
        drug_ids = drug_input.get_feature_matrix(view="drug_id", identifiers=output.drug_ids)
        drug_means = {}
        for drug_output, drug_feature in zip(unique(output.drug_ids), unique(drug_ids), strict=True):
            responses_drug = output.response[drug_feature == output.drug_ids]
            if len(responses_drug) > 0:
                drug_means[drug_output] = np.mean(responses_drug)

        # Compute the effects as deviations from the overall mean.
        self.cell_line_effects = {cl: (mean - self.dataset_mean) for cl, mean in cell_line_means.items()}
        self.drug_effects = {drug: (mean - self.dataset_mean) for drug, mean in drug_means.items()}

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts responses for given cell line and drug pairs.

        The prediction is computed as:
            prediction = overall_mean + cell_line_effect + drug_effect

        If a cell line or drug has not been seen during training, their effect is set to zero.

        :param cell_line_ids: Array of cell line IDs.
        :param drug_ids: Array of drug IDs.
        :param cell_line_input: Not used.
        :param drug_input: Not used.
        :return: NumPy array of predicted responses.
        """
        predictions = []
        for cl, drug in zip(cell_line_ids, drug_ids):
            effect_cl = self.cell_line_effects.get(cl, 0)
            effect_drug = self.drug_effects.get(drug, 0)
            # ANOVA-based prediction: overall mean + cell line effect + drug effect.
            predictions.append(self.dataset_mean + effect_cl + effect_drug)
        return np.array(predictions)

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
        Loads the drug features.

        :param data_path: Path to the data.
        :param dataset_name: Name of the dataset.
        :return: FeatureDataset containing the drug IDs.
        """
        return load_drug_ids_from_csv(data_path, dataset_name)
