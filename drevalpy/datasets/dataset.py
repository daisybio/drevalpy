# flake8: noqa: RST201, RST203, RST301, RST401

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
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.model_selection import GroupKFold, train_test_split

from ..pipeline_function import pipeline_function
from .utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER, permute_features, randomize_graph

np.set_printoptions(threshold=6)


class DrugResponseDataset:
    """Drug response dataset."""

    _response: np.ndarray
    _cell_line_ids: np.ndarray
    _tissues: np.ndarray | None = None
    _drug_ids: np.ndarray
    _predictions: np.ndarray | None = None
    _cv_splits: list[dict[str, "DrugResponseDataset"]] = []
    _name: str

    @pipeline_function
    def __init__(
        self,
        response: np.ndarray,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        tissues: np.ndarray | None = None,
        predictions: np.ndarray | None = None,
        dataset_name: str = "unnamed",
    ) -> None:
        """
        Initializes the drug response dataset.

        :param response: drug response values per cell line and drug
        :param cell_line_ids: cell line IDs
        :param drug_ids: drug IDs
        :param tissues: Optionally, tissue types of the cell lines for leave-tissue-out cv
        :param predictions: optional. Predicted drug response values per cell line and drug
        :param dataset_name: optional. Name of the dataset, default: "unnamed"
        :raises AssertionError: If response, cell_line_ids, drug_ids, (and the optional predictions) do not all have
            the same length.
        """
        super().__init__()
        if len(response) != len(cell_line_ids):
            raise AssertionError("Response and cell line identifiers have different lengths.")
        if len(response) != len(drug_ids):
            raise AssertionError("Response and drug identifiers have different lengths.")
        if predictions is not None and len(response) != len(predictions):
            raise AssertionError("Response and predictions have different lengths.")
        self._response = response
        self._cell_line_ids = cell_line_ids
        self._drug_ids = drug_ids
        self._predictions = predictions
        self._name = dataset_name
        if tissues is not None:
            self._tissues = np.array(tissues)
        else:
            self._tissues = None

    @pipeline_function
    @classmethod
    def from_csv(
        cls: type["DrugResponseDataset"],
        input_file: str | Path,
        dataset_name: str = "unknown",
        measure: str = "response",
        tissue_column: str | None = None,
    ) -> "DrugResponseDataset":
        """
        Load a dataset from a csv file.

        This function creates a DrugResponseDataset from a provided input file in csv format.
        The following columns are required:
        - response:         the drug response values as floating point values
        - cell_line_name:    a string identifier for cell lines
        - pubchem_id:         a string identifier for drugs
        - predictions:      an optional column containing drug response predictions
        - LN_IC50_curvecurator:         the name of the column containing the measure to predict

        :param input_file: Path to the csv file containing the data to be loaded
        :param dataset_name: Optional name to associate the dataset with, default = "unknown"
        :param measure: The name of the column containing the measure to predict, default = "response"
        :param tissue_column: Optional column name of column containing tissue types
        :raises ValueError: If the required columns are not found in the input file
        :returns: DrugResponseDataset object containing data from provided csv file.
        """
        data = pd.read_csv(input_file)

        if measure not in data.columns:
            raise ValueError(f"Column {measure} not found in the input file.")
        elif CELL_LINE_IDENTIFIER not in data.columns:
            raise ValueError(f"Column {CELL_LINE_IDENTIFIER} not found in the input file.")
        elif DRUG_IDENTIFIER not in data.columns:
            raise ValueError(f"Column {DRUG_IDENTIFIER} not found in the input file.")

        data[DRUG_IDENTIFIER] = data[DRUG_IDENTIFIER].astype(str)
        if "predictions" in data.columns:
            predictions = data["predictions"].values
        else:
            predictions = None
        return cls(
            response=data[measure].values,
            cell_line_ids=data[CELL_LINE_IDENTIFIER].values,
            drug_ids=data[DRUG_IDENTIFIER].values,
            predictions=predictions,
            dataset_name=dataset_name,
            tissues=data[tissue_column].values if tissue_column in data.columns else None,
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
    def tissue(self) -> np.ndarray | None:
        """
        Returns the tissue types if they exist.

        :returns: numpy array containing tissue types or None.
        """
        return self._tissues

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

        Used in the pipeline.

        :returns: dataset name.
        """
        return self._name

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

        :returns: pandas DataFrame of the dataset)
        """
        data = {
            CELL_LINE_IDENTIFIER: self.cell_line_ids,
            DRUG_IDENTIFIER: self.drug_ids,
            "response": self.response,
        }
        if self.predictions is not None:
            data["predictions"] = self.predictions
        if self.tissue is not None:
            data["tissue"] = self.tissue

        return pd.DataFrame(data)

    def to_csv(self, path: str | Path):
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

        if self.tissue is not None and other.tissue is not None:
            self._tissues = np.concatenate([self.tissue, other.tissue])

        if self.predictions is not None and other.predictions is not None:
            self._predictions = np.concatenate([self._predictions, other.predictions])

    @pipeline_function
    def remove_nan_responses(self) -> None:
        """Removes rows with NaN values in the response."""
        mask = ~np.isnan(self.response)
        self.mask(mask)

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
        if self.tissue is not None:
            self._tissues = self.tissue[indices]

    def _remove_drugs(self, drugs_to_remove: str | list[str | int]) -> None:
        """
        Removes drugs from the dataset.

        :param drugs_to_remove: name of drug or list of names of multiple drugs to remove
        """
        if isinstance(drugs_to_remove, str):
            drugs_to_remove = [drugs_to_remove]

        mask = np.array([drug not in drugs_to_remove for drug in self.drug_ids], dtype=bool)
        self.mask(mask)

    def _remove_cell_lines(self, cell_lines_to_remove: str | list[str | int]) -> None:
        """
        Removes cell lines from the dataset.

        :param cell_lines_to_remove: name of cell line or list of names of multiple cell lines to remove
        """
        if isinstance(cell_lines_to_remove, str):
            cell_lines_to_remove = [cell_lines_to_remove]

        mask = np.array([cell_line not in cell_lines_to_remove for cell_line in self.cell_line_ids], dtype=bool)
        self.mask(mask)

    def remove_rows(self, indices: np.ndarray) -> None:
        """
        Removes rows from the dataset.

        :param indices: indices of rows to remove
        """
        mask = np.ones(len(self), dtype=bool)
        mask[indices] = False
        self.mask(mask)

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
        :raises ValueError: if LTO cross-validation but tissue information not provided
        """
        if mode == "LPO":
            cv_splits = _leave_pair_out_cv(
                n_cv_splits=n_cv_splits,
                response=self.response,
                cell_line_ids=self.cell_line_ids,
                drug_ids=self.drug_ids,
                tissues=self.tissue,
                split_validation=split_validation,
                validation_ratio=validation_ratio,
                random_state=random_state,
                dataset_name=self.dataset_name,
            )

        elif mode in ["LCO", "LTO", "LDO"]:
            if mode == "LTO":
                # Leave-tissue-out cross-validation
                group = "tissue"
                if self.tissue is None:
                    raise ValueError("Tissue information is required for LTO cross-validation.")
            elif mode == "LCO":
                # Leave-cell-line-out cross-validation
                group = "cell_line"
            else:
                # Leave-drug-out cross-validation
                group = "drug"

            cv_splits = _leave_group_out_cv(
                group=group,
                n_cv_splits=n_cv_splits,
                response=self.response,
                cell_line_ids=self.cell_line_ids,
                drug_ids=self.drug_ids,
                tissues=self.tissue,
                split_validation=split_validation,
                validation_ratio=validation_ratio,
                random_state=random_state,
                dataset_name=self.dataset_name,
            )
        else:
            raise ValueError(f"Unknown split mode {mode!r}. Choose from 'LPO', 'LCO', 'LTO', 'LDO'.")

        if split_validation and split_early_stopping:
            for split in cv_splits:
                validation_es, early_stopping = split_early_stopping_data(split["validation"], test_mode=mode)
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
                    split[mode].to_csv(path=split_path)

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
            tissues=copy.deepcopy(self.tissue),
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
                (tuple(self.tissue) if self.tissue is not None else None),
            )
        )

    def mask(self, mask: np.ndarray) -> None:
        """
        Removes rows from the dataset based on a boolean mask.

        :param mask: boolean mask
        :raises ValueError: if mask is not boolean or integer
        """
        if mask.dtype != bool and not np.issubdtype(mask.dtype, np.integer):
            raise ValueError("Mask must be of boolean or integer dtype.")

        self._response = self.response[mask]
        self._cell_line_ids = self.cell_line_ids[mask]
        self._drug_ids = self.drug_ids[mask]
        if self.predictions is not None:
            self._predictions = self.predictions[mask]
        if self.tissue is not None:
            self._tissues = self.tissue[mask]

    @pipeline_function
    def transform(self, response_transformation: TransformerMixin) -> None:
        """
        Apply transformation to the response data and prediction data of the dataset.

        :param response_transformation: e.g., StandardScaler, MinMaxScaler, RobustScaler
        """
        self._response = response_transformation.transform(self.response.reshape(-1, 1)).squeeze()
        if self.predictions is not None:
            self._predictions = response_transformation.transform(self.predictions.reshape(-1, 1)).squeeze()

    @pipeline_function
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


@pipeline_function
def split_early_stopping_data(
    validation_dataset: DrugResponseDataset, test_mode: str
) -> tuple[DrugResponseDataset, DrugResponseDataset]:
    """
    Splits the validation dataset into a validation and an early stopping dataset.

    :param validation_dataset: input validation dataset
    :param test_mode: LPO, LCO, LTO, LDO
    :raises ValueError: if test_mode is not one of the expected values
    :returns: the resulting validation and early stopping datasets
    """
    validation_dataset.shuffle(random_state=42)

    # Determine the number of splits b (default 4,
    # but can be less if there are not enough groups)
    if test_mode == "LTO":
        tissues = validation_dataset.tissue
        if tissues is None:
            raise ValueError("Tissue information is required for LTO.")
        n_splits = min(4, len(np.unique(tissues)))
    elif test_mode == "LCO":
        n_splits = min(4, len(np.unique(validation_dataset.cell_line_ids)))
    elif test_mode == "LDO":
        n_splits = min(4, len(np.unique(validation_dataset.drug_ids)))
    else:
        n_splits = 4

    cv_v = validation_dataset.split_dataset(
        n_cv_splits=n_splits,
        mode=test_mode,
        split_validation=False,
        split_early_stopping=False,
        random_state=42,
    )
    # take the first fold of a 4 cv as the split i.e. 3/4 for validation and 1/4 for early stopping
    # when n_groups is less than 4, we splits the validation dataset into 2/3 and 1/3 or 1/2 and 1/2
    validation_dataset = cv_v[0]["train"]
    early_stopping_dataset = cv_v[0]["test"]
    return validation_dataset, early_stopping_dataset


def _leave_pair_out_cv(
    n_cv_splits: int,
    response: np.ndarray,
    cell_line_ids: np.ndarray,
    drug_ids: np.ndarray,
    tissues: np.ndarray | None = None,
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
    :param tissues: tissue types of the cell line, if available
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
    if tissues is not None:
        tissues = tissues[shuffled_indices].copy()

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
                tissues=tissues[train_indices] if tissues is not None else None,
                dataset_name=dataset_name,
            ),
            "test": DrugResponseDataset(
                cell_line_ids=cell_line_ids[test_indices],
                drug_ids=drug_ids[test_indices],
                response=response[test_indices],
                tissues=tissues[test_indices] if tissues is not None else None,
                dataset_name=dataset_name,
            ),
        }

        if split_validation:
            cv_fold["validation"] = DrugResponseDataset(
                cell_line_ids=cell_line_ids[validation_indices],
                drug_ids=drug_ids[validation_indices],
                response=response[validation_indices],
                tissues=tissues[validation_indices] if tissues is not None else None,
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
    tissues: np.ndarray | None = None,
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
    :param tissues: tissue types of the cell line, if available (required for LTO)
    :param split_validation: whether to split the training set into training and validation set
    :param validation_ratio: ratio of validation set (of the training set)
    :param random_state: random state
    :param dataset_name: name of the dataset
    :returns: list of dicts of the cross validation sets
    :raises AssertionError: if group is not 'cell_line' or 'drug' or 'tissue'
    :raises AssertionError: Tissue information is required for LTO cross-validation
    """
    if group not in {"cell_line", "drug", "tissue"}:
        raise AssertionError(f"group must be 'cell_line' or 'drug', but is {group}")

    if group == "cell_line":
        group_ids = cell_line_ids
    elif group == "drug":
        group_ids = drug_ids
    elif group == "tissue":
        if tissues is None:
            raise AssertionError("Tissue information is required for LTO cross-validation.")
        group_ids = tissues
    else:
        raise AssertionError(f"Unknown group {group}")

    # shuffle, since GroupKFold does not implement this
    indices = np.arange(len(response))
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(indices)
    response = response[shuffled_indices].copy()
    cell_line_ids = cell_line_ids[shuffled_indices].copy()
    drug_ids = drug_ids[shuffled_indices].copy()
    tissues = tissues[shuffled_indices].copy() if tissues is not None else None
    group_ids = group_ids[shuffled_indices].copy()
    gkf = GroupKFold(n_splits=n_cv_splits)
    cv_sets = []

    for train_indices, test_indices in gkf.split(response, groups=group_ids):
        cv_fold = {
            "train": DrugResponseDataset(
                cell_line_ids=cell_line_ids[train_indices],
                drug_ids=drug_ids[train_indices],
                response=response[train_indices],
                tissues=tissues[train_indices] if tissues is not None else None,
                dataset_name=dataset_name,
            ),
            "test": DrugResponseDataset(
                cell_line_ids=cell_line_ids[test_indices],
                drug_ids=drug_ids[test_indices],
                response=response[test_indices],
                tissues=tissues[test_indices] if tissues is not None else None,
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
                tissues=tissues[train_indices] if tissues is not None else None,
                dataset_name=dataset_name,
            )
            cv_fold["validation"] = DrugResponseDataset(
                cell_line_ids=cell_line_ids[validation_indices],
                drug_ids=drug_ids[validation_indices],
                response=response[validation_indices],
                tissues=tissues[validation_indices] if tissues is not None else None,
                dataset_name=dataset_name,
            )

        cv_sets.append(cv_fold)
    return cv_sets


class FeatureDataset:
    """
    Class for feature datasets.

    This class represents datasets with one or more views of features associated with a set of entities,
    such as drugs or cell lines. The feature data is stored in a nested dictionary structure:

    {
        identifier_1: {
            view_name_1: feature_vector,
            view_name_2: feature_vector,
            ...
        },
        identifier_2: {
            view_name_1: feature_vector,
            view_name_2: feature_vector,
            ...
        },
        ...
    }

    - Each outer key is a string identifier (e.g. a cell line ID or drug ID)
    - Each inner key is the name of a view (e.g. 'gene_expression', 'fingerprints')
    - Each inner value is a feature vector or object representing that view for the identifier
    """

    _features: dict[str, dict[str, Any]]
    _meta_info: dict[str, Any]

    @classmethod
    def from_csv(
        cls: type["FeatureDataset"],
        path_to_csv: str | Path,
        id_column: str,
        view_name: str,
        drop_columns: list[str] | None = None,
        transpose: bool = False,
        extract_meta_info: bool = True,
    ):
        """Load a one-view feature dataset from a csv file.

        Load a feature dataset from a csv file. The rows of the csv file represent the instances (cell lines or drugs),
        the columns represent the features. A column named id_column contains the identifiers of the instances.
        All unrelated columns (e.g. other id columns) should be provided as drop_columns,
        that will be removed from the dataset.

        :param path_to_csv: path to the csv file containing the data to be loaded
        :param view_name: name of the view (e.g. gene_expression)
        :param id_column: name of the column containing the identifiers
        :param drop_columns: list of columns to drop (e.g. other identifier columns)
        :param transpose: if True, the csv is transposed, i.e. the rows become columns and vice versa
        :param extract_meta_info: if True, extracts meta information from the dataset, e.g. gene names for gene expression
        :returns: FeatureDataset object containing data from provided csv file.
        """
        data = pd.read_csv(path_to_csv).T if transpose else pd.read_csv(path_to_csv)
        data[id_column] = data[id_column].astype(str)
        ids = data[id_column].values
        data_features = data.drop(columns=(drop_columns or []))
        data_features = data_features.set_index(id_column)
        data_features = data_features[~data_features.index.duplicated(keep="first")]
        features = {}

        for identifier in ids:
            features_for_instance = data_features.loc[identifier].values
            features[identifier] = {view_name: features_for_instance}

        meta_info = {}
        if extract_meta_info:
            meta_info = {view_name: list(data_features.columns)}

        return cls(features=features, meta_info=meta_info)

    def to_csv(self, path: str | Path, id_column: str, view_name: str):
        """
        Save the feature dataset to a CSV file. If meta_info is available for the view and valid,
        it will be written as column names.

        :param path: Path to the CSV file.
        :param id_column: Name of the column containing the identifiers.
        :param view_name: Name of the view.
        """
        data = []
        feature_names = None

        for identifier, feature_dict in self.features.items():
            vector = feature_dict.get(view_name)
            if vector is None:
                raise ValueError(f"View {view_name!r} not found for identifier {identifier!r}.")

            if feature_names is None:
                meta_names = self.meta_info.get(view_name)
                if isinstance(meta_names, list) and len(meta_names) == len(vector):
                    feature_names = meta_names
                else:
                    feature_names = [f"feature_{i}" for i in range(len(vector))]

            row = {id_column: identifier}
            row.update({name: value for name, value in zip(feature_names, vector)})
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

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

        Used in the pipeline.

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
        self._meta_info = meta_info if meta_info is not None else {}
        if meta_info is not None:
            # assert that str of meta Dict[str, Any] is in view_names
            if not all(meta_key in self.view_names for meta_key in meta_info.keys()):
                raise AssertionError(f"Meta keys {meta_info.keys()} not in view names {self.view_names}")
            self._meta_info = meta_info

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
        return FeatureDataset(features=copy.deepcopy(self.features), meta_info=copy.deepcopy(self.meta_info))

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
            print(f"Train IDs: {train_ids}")

            raise AssertionError("Train IDs should be unique.")

        train_features = np.vstack([self.features[identifier][view] for identifier in train_ids])
        transformer.fit(train_features)

        # Apply transformation and scaling to each feature vector
        for identifier in self.features:
            feature_vector = self.features[identifier][view]
            transformed_vector = transformer.transform([feature_vector])[0]
            self.features[identifier][view] = transformed_vector
        return transformer

    def apply(self, function: Callable, view: str):
        """Applies a function to the features of a view.

        :param function: function to apply
        :param view: view to apply the function to
        """
        for identifier in self.features:
            self.features[identifier][view] = function(self.features[identifier][view])
