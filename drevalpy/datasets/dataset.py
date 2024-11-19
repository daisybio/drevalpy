"""
Defines the different dataset classes.

DrugResponseDataset for response values and FeatureDataset for feature values.
They both inherit from the abstract class Dataset.
The DrugResponseDataset class is used
to store drug response values per cell line and drug.
The FeatureDataset class is used to store
feature values per cell line or drug.
The FeatureDataset class can also store meta information
for the feature views. The DrugResponseDataset class
can be split into training, validation and test sets for cross-validation.
The FeatureDataset class can be used to randomize feature vectors.
"""

import copy
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.model_selection import GroupKFold, train_test_split

from ..pipeline_function import pipeline_function
from .utils import permute_features, randomize_graph

np.set_printoptions(threshold=6)


class Dataset(ABC):
    """Abstract wrapper class for datasets."""

    @classmethod
    @abstractmethod
    def from_csv(cls: type["Dataset"], input_file: str | Path, dataset_name: str = "unknown") -> "Dataset":
        """
        Loads the dataset from data.

        :param input_file: Path to the csv file containing the data to be loaded
        :param dataset_name: Optional name to associate the dataset with, default = "unknown"
        :returns: Dataset object containing data from provided csv file.
        """

    @abstractmethod
    def save(self, path: str):
        """
        Saves the dataset to data.

        :param path: path to the dataset
        """


class DrugResponseDataset(Dataset):
    """Drug response dataset."""

    _response: np.ndarray
    _cell_line_ids: np.ndarray
    _drug_ids: np.ndarray
    _predictions: np.ndarray | None = None
    _cv_splits: list[dict[str, "DrugResponseDataset"]] = []
    _name: str

    @classmethod
    def from_csv(
        cls: type["DrugResponseDataset"], input_file: str | Path, dataset_name: str = "unknown"
    ) -> "DrugResponseDataset":
        """
        Load a dataset from a csv file.

        This function creates a DrugResponseDataset from a provided input file in csv format.
        The following columns are required:
        - response:         the drug response values as floating point values
        - cell_line_ids:    a string identifier for cell lines
        - drug_ids:         a string identifier for drugs
        - predictions:      an optional column containing a predicted value TODO what exactly?

        :param input_file: Path to the csv file containing the data to be loaded
        :param dataset_name: Optional name to associate the dataset with, default = "unknown"

        :returns: DrugResponseDataset object containing data from provided csv file.
        """
        data = pd.read_csv(input_file)
        if "predictions" in data.columns:
            predictions = data["predictions"].values
        else:
            predictions = None
        return cls(
            response=data["response"].values,
            cell_line_ids=data["cell_line_id"].values,
            drug_ids=data["drug_id"].values,
            predictions=predictions,
            dataset_name=dataset_name,
        )

    @property
    def response(self) -> np.ndarray:
        """
        Returns the response values.

        :returns: numpy array containing response values.
        """
        return self._response

    @property
    def cell_line_ids(self) -> np.ndarray:
        """
        Returns the cell_line_ids.

        :returns: numpy array containing cell_line_ids values.
        """
        return self._cell_line_ids

    @property
    def drug_ids(self) -> np.ndarray:
        """
        Returns the drug_ids.

        :returns: numpy array containing drug_ids values.
        """
        return self._drug_ids

    @property
    def predictions(self) -> np.ndarray | None:
        """
        Returns the predictions if they exist.

        :returns: numpy array containing prediction values or None.
        """
        return self._predictions

    @property
    def cv_splits(self) -> list[dict[str, "DrugResponseDataset"]]:
        """
        Returns the cv_splits.

        :returns: DrugResponseDatasets containing the CV_splits.
        """
        return self._cv_splits

    @property
    def dataset_name(self) -> str:
        """
        Returns the name of this DrugResponseDataset.

        :returns: dataset name.
        """
        return self._name

    @pipeline_function
    def __init__(
        self,
        response: np.ndarray,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        predictions: np.ndarray | None = None,
        dataset_name: str = "unnamed",
    ) -> None:
        """
        Initializes the drug response dataset.

        :param response: drug response values per cell line and drug
        :param cell_line_ids: cell line IDs
        :param drug_ids: drug IDs
        :param predictions: optional. Predicted drug response values per cell line and drug
        :param dataset_name: optional. Name of the dataset, default: "unnamed"
        :raises AssertionError: If response, cell_line_ids, drug_ids, (and the optional predictions) do not all have
            the same length.
        """
        super().__init__()
        if len(response) != len(cell_line_ids):
            raise AssertionError("Response and cell_line_ids have different lengths.")
        if len(response) != len(drug_ids):
            raise AssertionError("Response and drug_ids have different lengths.")
        if predictions is not None and len(response) != len(predictions):
            raise AssertionError("Response and predictions have different lengths.")
        self._response = response
        self._cell_line_ids = cell_line_ids
        self._drug_ids = drug_ids
        self._predictions = predictions
        self._name = dataset_name

    def __len__(self) -> int:
        """
        Overwrites the default length method.

        :returns: Number of samples in the dataset
        """
        return len(self.response)

    def __str__(self) -> str:
        """
        Overwrite the default str method.

        :return: Text summary of the dataset
        """
        string = (
            f"{self.dataset_name} DrugResponseDataset with {len(self)} entries:\n"
            f"CLs {self.cell_line_ids}\n"
            f"Drugs {self.drug_ids}\n"
            f"Response {self.response}\n"
        )
        if self.predictions is not None:
            string += f"Predictions {self.predictions}\n"
        return string

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the dataset into a pandas DataFrame.

        :returns: pandas DataFrame of the dataset with columns 'cell_line_id', 'drug_id', 'response'(, 'predictions')
        """
        data = {
            "cell_line_id": self.cell_line_ids,
            "drug_id": self.drug_ids,
            "response": self.response,
        }
        if self.predictions is not None:
            data["predictions"] = self.predictions
        return pd.DataFrame(data)

    def save(self, path: str | Path):
        """
        Stores the drug response dataset on disk.

        :param path: path to desired storage location
        """
        self.to_dataframe().to_csv(path, index=False)

    @pipeline_function
    def add_rows(self, other: "DrugResponseDataset") -> None:
        """
        Adds rows from another dataset.

        :param other: other dataset
        """
        self._response = np.concatenate([self._response, other.response])
        self._cell_line_ids = np.concatenate([self._cell_line_ids, other.cell_line_ids])
        self._drug_ids = np.concatenate([self._drug_ids, other.drug_ids])

        if self.predictions is not None and other.predictions is not None:
            self._predictions = np.concatenate([self._predictions, other.predictions])

    @pipeline_function
    def remove_nan_responses(self) -> None:
        """Removes rows with NaN values in the response."""
        mask = np.isnan(self.response)
        self._response = self.response[~mask]
        self._cell_line_ids = self.cell_line_ids[~mask]
        self._drug_ids = self.drug_ids[~mask]
        if self.predictions is not None:
            self._predictions = self.predictions[~mask]

    @pipeline_function
    def shuffle(self, random_state: int = 42) -> None:
        """
        Shuffles the dataset.

        :param random_state: random state
        """
        indices = np.arange(len(self))
        np.random.seed(random_state)
        np.random.shuffle(indices)
        self._response = self.response[indices]
        self._cell_line_ids = self.cell_line_ids[indices]
        self._drug_ids = self.drug_ids[indices]
        if self.predictions is not None:
            self._predictions = self.predictions[indices]

    def _remove_drugs(self, drugs_to_remove: str | list[str | int]) -> None:
        """
        Removes drugs from the dataset.

        :param drugs_to_remove: name of drug or list of names of multiple drugs to remove
        """
        if isinstance(drugs_to_remove, str):
            drugs_to_remove = [drugs_to_remove]

        mask = [drug not in drugs_to_remove for drug in self.drug_ids]
        self._drug_ids = self.drug_ids[mask]
        self._cell_line_ids = self.cell_line_ids[mask]
        self._response = self.response[mask]

    def _remove_cell_lines(self, cell_lines_to_remove: str | list[str | int]) -> None:
        """
        Removes cell lines from the dataset.

        :param cell_lines_to_remove: name of cell line or list of names of multiple cell lines to remove
        """
        if isinstance(cell_lines_to_remove, str):
            cell_lines_to_remove = [cell_lines_to_remove]

        mask = [cell_line not in cell_lines_to_remove for cell_line in self.cell_line_ids]
        self._drug_ids = self.drug_ids[mask]
        self._cell_line_ids = self.cell_line_ids[mask]
        self._response = self.response[mask]

    def remove_rows(self, indices: np.ndarray) -> None:
        """
        Removes rows from the dataset.

        :param indices: indices of rows to remove
        """
        indices = np.array(indices, dtype=int)
        self._drug_ids = np.delete(self.drug_ids, indices)
        self._cell_line_ids = np.delete(self.cell_line_ids, indices)
        self._response = np.delete(self.response, indices)
        if self.predictions is not None:
            self._predictions = np.delete(self.predictions, indices)

    def reduce_to(self, cell_line_ids: np.ndarray | None = None, drug_ids: np.ndarray | None = None) -> None:
        """
        Removes all rows which contain a cell_line not in cell_line_ids or a drug not in drug_ids.

        :param cell_line_ids: cell line IDs or None to keep all cell lines
        :param drug_ids: drug IDs or None to keep all cell lines
        """
        if drug_ids is not None:
            self._remove_drugs(list(set(self.drug_ids) - set(drug_ids)))

        if cell_line_ids is not None:
            self._remove_cell_lines(list(set(self.cell_line_ids) - set(cell_line_ids)))

    @pipeline_function
    def split_dataset(
        self,
        n_cv_splits: int,
        mode: str,
        split_validation: bool = True,
        split_early_stopping: bool = True,
        validation_ratio: float = 0.1,
        random_state: int = 42,
    ) -> list[dict]:
        """
        Splits the dataset into training, validation and test sets for cross-validation.

        :param n_cv_splits: number of cross-validation splits, e.g., 5
        :param mode: split mode ('LPO', 'LCO', 'LDO')
        :param split_validation: if True, a validation set is generated
        :param split_early_stopping: if True, an early stopping set is generated
        :param validation_ratio: ratio of validation set size to training set size
        :param random_state: random state
        :returns: list of dictionaries containing the cross-validation datasets.
            Each fold is a dictionary with keys 'train', 'validation', 'test', 'validation_es', 'early_stopping'.
        :raises ValueError: if mode is not 'LPO', 'LCO', or 'LDO'
        """
        cell_line_ids = self.cell_line_ids
        drug_ids = self.drug_ids
        response = self.response

        if mode == "LPO":
            cv_splits = _leave_pair_out_cv(
                n_cv_splits,
                response,
                cell_line_ids,
                drug_ids,
                split_validation,
                validation_ratio,
                random_state,
                self.dataset_name,
            )

        elif mode in ["LCO", "LDO"]:
            group = "cell_line" if mode == "LCO" else "drug"
            cv_splits = _leave_group_out_cv(
                group=group,
                n_cv_splits=n_cv_splits,
                response=response,
                cell_line_ids=cell_line_ids,
                drug_ids=drug_ids,
                split_validation=split_validation,
                validation_ratio=validation_ratio,
                random_state=random_state,
                dataset_name=self.dataset_name,
            )
        else:
            raise ValueError(f"Unknown split mode {mode!r}. Choose from 'LPO', 'LCO', 'LDO'.")

        if split_validation and split_early_stopping:
            for split in cv_splits:
                validation_es, early_stopping = _split_early_stopping_data(split["validation"], test_mode=mode)
                split["validation_es"] = validation_es
                split["early_stopping"] = early_stopping
        self._cv_splits = cv_splits
        return cv_splits

    def save_splits(self, path: str):
        """
        Save cross validation splits to path/cv_split_0_train.csv and path/cv_split_0_test.csv.

        :param path: path to the directory where the cv split files are saved
        :raises AssertionError: if DrugResponseDataset was not split
        """
        if not self.cv_splits:
            raise AssertionError("Trying to save splits, but DrugResponseDataset was not split.")
        os.makedirs(path, exist_ok=True)
        for i, split in enumerate(self.cv_splits):

            for mode in [
                "train",
                "validation",
                "test",
                "validation_es",
                "early_stopping",
            ]:
                if mode in split:
                    split_path = os.path.join(path, f"cv_split_{i}_{mode}.csv")
                    split[mode].save(path=split_path)

    def load_splits(self, path: str) -> None:
        """
        Load cross validation splits from path/cv_split_0_train.csv and path/cv_split_0_test.csv.

        :param path: path to the directory containing the cv split files
        :raises AssertionError: if no cv split files are found in path
        """
        files = os.listdir(path)
        files = [file for file in files if (file.endswith(".csv") and file.startswith("cv_split"))]
        if len(files) == 0:
            raise AssertionError(f"No cv split files found in {path}")

        train_splits = [file for file in files if "train" in file]
        test_splits = [file for file in files if "test" in file]

        validation_es_splits = [file for file in files if "validation_es" in file]
        validation_splits = [file for file in files if "validation" in file and file not in validation_es_splits]
        early_stopping_splits = [file for file in files if "early_stopping" in file]

        for ds in [
            train_splits,
            test_splits,
            validation_splits,
            validation_es_splits,
            early_stopping_splits,
        ]:
            ds.sort()

        optional_splits = {
            "validation": validation_splits,
            "validation_es": validation_es_splits,
            "early_stopping": early_stopping_splits,
        }
        self._cv_splits.clear()  # TODO do we need this?

        for split_train, split_test in zip(train_splits, test_splits, strict=True):
            tr_split = DrugResponseDataset.from_csv(os.path.join(path, split_train), dataset_name=self.dataset_name)
            te_split = DrugResponseDataset.from_csv(os.path.join(path, split_test), dataset_name=self.dataset_name)
            self._cv_splits.append({"train": tr_split, "test": te_split})

        for mode in ["validation", "validation_es", "early_stopping"]:
            if len(optional_splits[mode]) > 0:
                for i, v_split in enumerate(optional_splits[mode]):
                    split = DrugResponseDataset.from_csv(os.path.join(path, v_split), dataset_name=self.dataset_name)
                    self._cv_splits[i][mode] = split

    def copy(self):
        """Returns a copy of the drug response dataset.

        :returns: copy of the dataset
        """
        return DrugResponseDataset(
            response=copy.deepcopy(self.response),
            cell_line_ids=copy.deepcopy(self.cell_line_ids),
            drug_ids=copy.deepcopy(self.drug_ids),
            predictions=copy.deepcopy(self.predictions),
            dataset_name=self.dataset_name,
        )

    def __hash__(self) -> int:
        """Overwrites default hash method.

        :returns: hash value of the dataset
        """
        return hash(
            (
                self.dataset_name,
                tuple(self.cell_line_ids),
                tuple(self.drug_ids),
                tuple(self.response),
                (tuple(self.predictions) if self.predictions is not None else None),
            )
        )

    def mask(self, mask: np.ndarray) -> None:
        """
        Removes rows from the dataset based on a boolean mask.

        :param mask: boolean mask
        """
        self._response = self.response[mask]
        self._cell_line_ids = self.cell_line_ids[mask]
        self._drug_ids = self.drug_ids[mask]
        if self.predictions is not None:
            self._predictions = self.predictions[mask]

    def transform(self, response_transformation: TransformerMixin) -> None:
        """
        Apply transformation to the response data and prediction data of the dataset.

        :param response_transformation: e.g., StandardScaler, MinMaxScaler, RobustScaler
        """
        self._response = response_transformation.transform(self.response.reshape(-1, 1)).squeeze()
        if self.predictions is not None:
            self._predictions = response_transformation.transform(self.predictions.reshape(-1, 1)).squeeze()

    def fit_transform(self, response_transformation: TransformerMixin) -> None:
        """
        Fit and transform the response data and prediction data of the dataset.

        :param response_transformation: e.g., StandardScaler, MinMaxScaler, RobustScaler
        """
        response_transformation.fit(self.response.reshape(-1, 1))
        self.transform(response_transformation)

    def inverse_transform(self, response_transformation: TransformerMixin) -> None:
        """
        Inverse transform the response data and prediction data of the dataset.

        :param response_transformation: e.g., StandardScaler, MinMaxScaler, RobustScaler
        """
        self._response = response_transformation.inverse_transform(self.response.reshape(-1, 1)).squeeze()
        if self.predictions is not None:
            self._predictions = response_transformation.inverse_transform(self.predictions.reshape(-1, 1)).squeeze()


def _split_early_stopping_data(
    validation_dataset: DrugResponseDataset, test_mode: str
) -> tuple[DrugResponseDataset, DrugResponseDataset]:
    """
    Splits the validation dataset into a validation and an early stopping dataset.

    :param validation_dataset: input validation dataset
    :param test_mode: LCO, LDO, LPO
    :returns: the resulting validation and early stopping datasets
    """
    validation_dataset.shuffle(random_state=42)
    cv_v = validation_dataset.split_dataset(
        n_cv_splits=4,
        mode=test_mode,
        split_validation=False,
        split_early_stopping=False,
        random_state=42,
    )
    # take the first fold of a 4 cv as the split i.e. 3/4 for validation and 1/4 for early stopping
    validation_dataset = cv_v[0]["train"]
    early_stopping_dataset = cv_v[0]["test"]
    return validation_dataset, early_stopping_dataset


def _leave_pair_out_cv(
    n_cv_splits: int,
    response: np.ndarray,
    cell_line_ids: np.ndarray,
    drug_ids: np.ndarray,
    split_validation: bool = True,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    dataset_name: str = "unknown",
) -> list[dict[str, DrugResponseDataset]]:
    """
    Leave pair out cross validation. Splits data into n_cv_splits number of cross validation splits.

    :param n_cv_splits: number of cross validation splits
    :param response: response (e.g. ic50 values)
    :param cell_line_ids: cell line IDs
    :param drug_ids: drug IDs
    :param split_validation: whether to split the training set into training and validation set
    :param validation_ratio: ratio of validation set (of the training set)
    :param random_state: random state
    :param dataset_name: name of the dataset
    :returns: list of dicts of the cross validation sets
    :raises AssertionError: if response, cell_line_ids and drug_ids have different lengths
    """
    if not (len(response) == len(cell_line_ids) == len(drug_ids)):
        raise AssertionError("response, cell_line_ids and drug_ids must have the same length")
    indices = np.arange(len(response))
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(indices)
    response = response[shuffled_indices].copy()
    cell_line_ids = cell_line_ids[shuffled_indices].copy()
    drug_ids = drug_ids[shuffled_indices].copy()

    # We use GroupKFold to ensure that each pair is only in one fold (prevent data leakage due to
    # experimental replicates).
    # If there are no replicates this is equivalent to KFold.
    groups = [cell + "_" + drug for cell, drug in zip(cell_line_ids, drug_ids, strict=True)]
    kf = GroupKFold(n_splits=n_cv_splits)
    cv_sets = []

    for train_indices, test_indices in kf.split(response, groups=groups):
        if split_validation:
            # split training set into training and validation set
            train_indices, validation_indices = train_test_split(
                train_indices,
                test_size=validation_ratio,
                shuffle=True,
                random_state=random_state,
            )
        cv_fold = {
            "train": DrugResponseDataset(
                cell_line_ids=cell_line_ids[train_indices],
                drug_ids=drug_ids[train_indices],
                response=response[train_indices],
                dataset_name=dataset_name,
            ),
            "test": DrugResponseDataset(
                cell_line_ids=cell_line_ids[test_indices],
                drug_ids=drug_ids[test_indices],
                response=response[test_indices],
                dataset_name=dataset_name,
            ),
        }

        if split_validation:
            cv_fold["validation"] = DrugResponseDataset(
                cell_line_ids=cell_line_ids[validation_indices],
                drug_ids=drug_ids[validation_indices],
                response=response[validation_indices],
                dataset_name=dataset_name,
            )

        cv_sets.append(cv_fold)
    return cv_sets


def _leave_group_out_cv(
    group: str,
    n_cv_splits: int,
    response: np.ndarray,
    cell_line_ids: np.ndarray,
    drug_ids: np.ndarray,
    split_validation: bool = True,
    validation_ratio: float = 0.1,
    random_state: int = 42,
    dataset_name: str = "unknown",
):
    """
    Leave group out cross validation: Splits data into n_cv_splits number of cross validation splits.

    :param group: group to leave out (cell_line or drug)
    :param n_cv_splits: number of cross validation splits
    :param response: response (e.g. ic50 values)
    :param cell_line_ids: cell line IDs
    :param drug_ids: drug IDs
    :param split_validation: whether to split the training set into training and validation set
    :param validation_ratio: ratio of validation set (of the training set)
    :param random_state: random state
    :param dataset_name: name of the dataset
    :returns: list of dicts of the cross validation sets
    :raises AssertionError: if group is not 'cell_line' or 'drug'
    """
    if group not in {"cell_line", "drug"}:
        raise AssertionError(f"group must be 'cell_line' or 'drug', but is {group}")

    if group == "cell_line":
        group_ids = cell_line_ids
    else:
        group_ids = drug_ids

    # shuffle, since GroupKFold does not implement this
    indices = np.arange(len(response))
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(indices)
    response = response[shuffled_indices].copy()
    cell_line_ids = cell_line_ids[shuffled_indices].copy()
    drug_ids = drug_ids[shuffled_indices].copy()
    group_ids = group_ids[shuffled_indices].copy()
    gkf = GroupKFold(n_splits=n_cv_splits)
    cv_sets = []

    for train_indices, test_indices in gkf.split(response, groups=group_ids):
        cv_fold = {
            "train": DrugResponseDataset(
                cell_line_ids=cell_line_ids[train_indices],
                drug_ids=drug_ids[train_indices],
                response=response[train_indices],
                dataset_name=dataset_name,
            ),
            "test": DrugResponseDataset(
                cell_line_ids=cell_line_ids[test_indices],
                drug_ids=drug_ids[test_indices],
                response=response[test_indices],
                dataset_name=dataset_name,
            ),
        }
        if split_validation:
            # split training set into training and validation set.
            # The validation set also does
            # contain unqiue cell lines/drugs
            unique_train_groups = np.unique(group_ids[train_indices])
            train_groups, validation_groups = train_test_split(
                unique_train_groups,
                test_size=validation_ratio,
                shuffle=True,
                random_state=random_state,
            )
            train_indices = np.where(np.isin(group_ids, train_groups))[0]
            validation_indices = np.where(np.isin(group_ids, validation_groups))[0]
            cv_fold["train"] = DrugResponseDataset(
                cell_line_ids=cell_line_ids[train_indices],
                drug_ids=drug_ids[train_indices],
                response=response[train_indices],
                dataset_name=dataset_name,
            )
            cv_fold["validation"] = DrugResponseDataset(
                cell_line_ids=cell_line_ids[validation_indices],
                drug_ids=drug_ids[validation_indices],
                response=response[validation_indices],
                dataset_name=dataset_name,
            )

        cv_sets.append(cv_fold)
    return cv_sets


class FeatureDataset(Dataset):
    """Class for feature datasets."""

    _meta_info: dict[str, Any] = {}
    _features: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_csv(
        cls: type["FeatureDataset"], input_file: str | Path, dataset_name: str = "unknown"
    ) -> "FeatureDataset":
        """
        Load a feature dataset from a csv file.

        This function creates a FeatureDataset from a provided input file in csv format.
        :param input_file: Path to the csv file containing the data to be loaded
        :param dataset_name: Optional name to associate the dataset with, default = "unknown"
        :raises NotImplementedError: This method is currently not implemented.
        """
        raise NotImplementedError

    @property
    def meta_info(self) -> dict[str, Any]:
        """
        Returns the meta information.

        :returns: Meta information of this FeatureDataset
        """
        return self._meta_info

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        """
        Returns the features.

        :returns: features of this FeatureDataset
        """
        return self._features

    @property
    def identifiers(self) -> np.ndarray:
        """
        Returns the identifiers of the features.

        :returns: feature identifiers of this FeatureDataset
        """
        return np.array(list(self.features.keys()))

    @property
    def view_names(self) -> list[str]:
        """
        Returns the view_names.

        :returns: view_names of this FeatureDataset
        """
        return list(self.features[list(self.features.keys())[0]].keys())  # TODO whut?!

    def __init__(
        self,
        features: dict[str, dict[str, Any]],
        meta_info: dict[str, Any] | None = None,
    ):
        """
        Initializes the feature dataset.

        :param features: dictionary of features,
            key: drug ID/cell line ID, value: Dict of feature views,
            key: feature name, value: feature vector
        :param meta_info: additional information for the views, e.g. gene names for gene expression
        :raises AssertionError: if meta_info keys are not in view names
        """
        super().__init__()
        self._features = features
        if meta_info is not None:
            # assert that str of meta Dict[str, Any] is in view_names
            if not all(meta_key in self.view_names for meta_key in meta_info.keys()):
                raise AssertionError(f"Meta keys {meta_info.keys()} not in view names {self.view_names}")
            self._meta_info = meta_info

    def save(self, path: str):
        """
        Saves the feature dataset to data.

        :param path: path to the dataset
        :raises NotImplementedError: if method is not implemented
        """
        raise NotImplementedError("save method not implemented")

    def randomize_features(self, views_to_randomize: str | list[str], randomization_type: str) -> None:
        """
        Randomizes the feature vectors.

        Permutation permutes the feature vectors.
        Invariant means that the randomization is done in a way that a key characteristic of the feature is
        preserved. In case of matrices, this is the mean and standard deviation of the feature view for this
        instance, for networks it is the degree distribution.

        :param views_to_randomize: name of feature view or list of names of multiple feature views
            to randomize. The other views are not randomized.
        :param randomization_type: randomization type ('permutation', 'invariant').
        :raises AssertionError: if randomization_type is not 'permutation' or 'invariant'
        :raises ValueError: if no invariant randomization is available for the feature view type
        """
        if randomization_type not in ["permutation", "invariant"]:
            raise AssertionError(
                f"Unknown randomization type {randomization_type!r}. Choose from 'permutation', 'invariant'."
            )

        if isinstance(views_to_randomize, str):
            views_to_randomize = [views_to_randomize]

        if randomization_type == "permutation":
            # Permute the specified views for each entity (= cell line or drug)
            # E.g. each cell line gets the feature vector/graph/image...
            # of another cell line.
            # Drawn without replacement.
            self._features = permute_features(
                features=self.features,
                views_to_permute=views_to_randomize,
                identifiers=self.identifiers,
                all_views=self.view_names,
            )

        elif randomization_type == "invariant":
            # Invariant randomization:
            # Randomize the specified views for each entity in a way that
            # a key characteristic of the feature is preserved.
            # For vectors this is the mean and standard deviation the feature view,
            # for networks the degree distribution.
            for view in views_to_randomize:
                for identifier in self.identifiers:
                    if isinstance(self.features[identifier][view], np.ndarray):
                        new_features = np.random.normal(
                            self.features[identifier][view].mean(),
                            self.features[identifier][view].std(),
                            self.features[identifier][view].shape,
                        )
                    elif isinstance(self.features[identifier][view], nx.classes.graph.Graph):
                        new_features = randomize_graph(self.features[identifier][view])

                    else:
                        raise ValueError(
                            f"No invariant randomization available for feature view "
                            f"type {type(self.features[identifier][view])!r}."
                        )
                    self.features[identifier][view] = new_features

    def get_feature_matrix(self, view: str, identifiers: np.ndarray) -> np.ndarray:
        """
        Returns the feature matrix for the given view.

        The feature view must be a vector or matrix.
        :param view: view name
        :param identifiers: list of identifiers (cell lines oder drugs)
        :returns: feature matrix
        :raises AssertionError: if no identifiers are given
        :raises AssertionError: if view is not in the FeatureDataset
        :raises AssertionError: if identifiers are not in the FeatureDataset
        :raises AssertionError: if feature vectors of view have different lengths
        :raises AssertionError: if view is not a numpy array, i.e. not a vector or matrix
        """
        if len(identifiers) == 0:
            raise AssertionError("get_feature_matrix: No identifiers given.")

        if view not in self.view_names:
            raise AssertionError(f"View {view!r} not in in the FeatureDataset.")
        missing_identifiers = {id_ for id_ in identifiers if id_ not in self.identifiers}
        if missing_identifiers:
            raise AssertionError(
                f"{len(missing_identifiers)} of {len(np.unique(identifiers))} ids are not in the "
                f"FeatureDataset. Missing ids: {missing_identifiers}"
            )

        if not all(len(self.features[id_][view]) == len(self.features[identifiers[0]][view]) for id_ in identifiers):
            raise AssertionError(f"Feature vectors of view {view} have different lengths.")

        if not all(isinstance(self.features[id_][view], np.ndarray) for id_ in identifiers):
            raise AssertionError(f"get_feature_matrix only works for vectors or matrices. {view} is not a numpy array.")
        out = np.array([self.features[id_][view] for id_ in identifiers])
        return out

    def copy(self):
        """Returns a copy of the feature dataset.

        :returns: copy of the dataset
        """
        return FeatureDataset(features=copy.deepcopy(self.features))

    def add_features(self, other: "FeatureDataset") -> None:
        """
        Adds features views from another dataset. Inner join (only common identifiers are kept).

        :param other: other dataset
        :raises AssertionError: if feature views overlap
        """
        if len(set(self.view_names) & set(other.view_names)) != 0:
            raise AssertionError(
                "Trying to add features but feature views overlap. FeatureDatasets should be distinct."
            )
        if other.meta_info:
            self.add_meta_info(other)

        common_identifiers = set(self.identifiers).intersection(other.identifiers)
        new_features = {}
        for id_ in common_identifiers:
            id_ = str(id_)
            new_features[id_] = {view: self.features[id_][view] for view in self.view_names}
            for view in other.view_names:
                new_features[id_][view] = other.features[id_][view]

        self._features = new_features

    def add_meta_info(self, other: "FeatureDataset") -> None:
        """
        Adds meta information to the feature dataset.

        :param other: other dataset
        """
        other_meta = other.meta_info
        if self.meta_info is None:
            self.meta_info = other_meta
        else:
            if other_meta is not None:
                self.meta_info.update(other_meta)

    def transform_features(self, ids: np.ndarray, transformer: TransformerMixin, view: str):
        """
        Applies a transformation like standard scaling to features.

        :param ids: The IDs to transform
        :param transformer: fitted sklearn transformer
        :param view: the view to transform
        :raises AssertionError: if view is not in the FeatureDataset
        :raises AssertionError: if a cell line is missing
        :raises AssertionError: if IDs are not unique
        """
        if view not in self.view_names:
            raise AssertionError(f"Transform view {view!r} not in in the FeatureDataset.")
        if not all([clid in self.features for clid in ids]):
            raise AssertionError("Trying to transform, but a cell line is missing.")

        if len(np.unique(ids)) != len(ids):
            raise AssertionError("IDs should be unique.")

        for identifier in ids:
            feature_vector = self.features[identifier][view]
            scaled_feature_vector = transformer.transform([feature_vector])[0]
            self.features[identifier][view] = scaled_feature_vector

    def fit_transform_features(self, train_ids: np.ndarray, transformer: TransformerMixin, view: str):
        """
        Fits and applies a transformation. Fitting is done only on the train_ids.

        :param train_ids: The IDs corresponding to the training dataset.
        :param transformer: sklearn transformer
        :param view: the view to transform
        :returns: The modified FeatureDataset with transformed gene expression features.
        :raises AssertionError: if view is not in the FeatureDataset
        :raises AssertionError: if train IDs are not unique
        """
        if view not in self.view_names:
            raise AssertionError(f"Transform view {view!r} not in in the FeatureDataset.")

        if len(np.unique(train_ids)) != len(train_ids):
            raise AssertionError("Train IDs should be unique.")

        # Collect all features of the view for fitting the scaler
        train_features = np.vstack([self.features[identifier][view] for identifier in train_ids])
        transformer.fit(train_features)

        # Apply transformation and scaling to each feature vector
        for identifier in self.features:
            feature_vector = self.features[identifier][view]
            scaled_gene_expression = transformer.transform([feature_vector])[0]
            self.features[identifier][view] = scaled_gene_expression
        return transformer

    def apply(self, function: Callable, view: str):
        """Applies a function to the features of a view.

        :param function: function to apply
        :param view: view to apply the function to
        """
        for identifier in self.features:
            self.features[identifier][view] = function(self.features[identifier][view])
