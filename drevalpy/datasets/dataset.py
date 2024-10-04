"""
Defines the different dataset classes: DrugResponseDataset for response values and
FeatureDataset for feature values. They both inherit from the abstract class Dataset.
The DrugResponseDataset class is used to store drug response values per cell line and drug.
The FeatureDataset class is used to store feature values per cell line or drug. The FeatureDataset
class can also store meta information for the feature views.
The DrugResponseDataset class can be split into training, validation and test sets for
cross-validation.
The FeatureDataset class can be used to randomize feature vectors.
"""

from abc import ABC, abstractmethod
import os
import copy
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.model_selection import GroupKFold, train_test_split
import networkx as nx

from .utils import (
    randomize_graph,
    permute_features,
)


class Dataset(ABC):
    """
    Abstract wrapper class for datasets.
    """

    @abstractmethod
    def load(self, path: str):
        """
        Loads the dataset from data.
        :param path: path to the dataset
        """

    @abstractmethod
    def save(self, path: str):
        """
        Saves the dataset to data.
        :param path: path to the dataset
        """


class DrugResponseDataset(Dataset):
    """
    Drug response dataset.
    """

    def __init__(
        self,
        response: Optional[ArrayLike] = None,
        cell_line_ids: Optional[ArrayLike] = None,
        drug_ids: Optional[ArrayLike] = None,
        predictions: Optional[ArrayLike] = None,
        dataset_name: Optional[str] = None,
    ):
        """
        Initializes the drug response dataset.
        :param response: drug response values per cell line and drug
        :param cell_line_ids: cell line IDs
        :param drug_ids: drug IDs
        :param predictions: optional. Predicted drug response values per cell line and drug
        :param dataset_name: optional. Name of the dataset
        """
        super().__init__()
        if response is not None:
            self.response = np.array(response)
            self.cell_line_ids = np.array(cell_line_ids)
            self.drug_ids = np.array(drug_ids)
            assert len(self.response) == len(
                self.cell_line_ids
            ), "response and cell_line_ids have different lengths"
            assert len(self.response) == len(
                self.drug_ids
            ), "response and drug_ids/cell_line_ids have different lengths"
            self.dataset_name = dataset_name
        else:
            self.response = response
            self.cell_line_ids = cell_line_ids
            self.drug_ids = drug_ids
            self.dataset_name = dataset_name

        if predictions is not None:
            self.predictions = np.array(predictions)
            assert len(self.predictions) == len(
                self.response
            ), "predictions and response have different lengths"
        else:
            self.predictions = None
        self.cv_splits = None

    def __len__(self):
        return len(self.response)

    def __str__(self):
        if len(self.response) > 3:
            string = (
                f"DrugResponseDataset: CLs {self.cell_line_ids[:3]}...; "
                f"Drugs {self.drug_ids[:3]}...; "
                f"Response {self.response[:3]}..."
            )
        else:
            string = (
                f"DrugResponseDataset: CLs {self.cell_line_ids}; "
                f"Drugs {self.drug_ids}; "
                f"Response {self.response}"
            )
        if self.predictions is not None:
            if len(self.predictions) > 3:
                string += f"; Predictions {self.predictions[:3]}..."
            else:
                string += f"; Predictions {self.predictions}"
        return string

    def to_dataframe(self):
        """
        Convert the dataset into a pandas DataFrame.
        """
        data = {
            "cell_line_id": self.cell_line_ids,
            "drug_id": self.drug_ids,
            "response": self.response,
        }
        if self.predictions is not None:
            data["predictions"] = self.predictions
        return pd.DataFrame(data)

    def load(self, path: str):
        """
        Loads the drug response dataset from data.
        :param path: path to the dataset
        """
        data = pd.read_csv(path)
        self.response = data["response"].values
        self.cell_line_ids = data["cell_line_ids"].values
        self.drug_ids = data["drug_ids"].values
        if "predictions" in data.columns:
            self.predictions = data["predictions"].values

    def save(self, path: str):
        """
        Saves the drug response dataset to data.
        :param path: path to the dataset
        """
        out = pd.DataFrame(
            {
                "cell_line_ids": self.cell_line_ids,
                "drug_ids": self.drug_ids,
                "response": self.response,
            }
        )
        if self.predictions is not None:
            out["predictions"] = self.predictions
        out.to_csv(path, index=False)

    def add_rows(self, other: "DrugResponseDataset") -> None:
        """
        Adds rows from another dataset.
        :param other: other dataset
        """
        self.response = np.concatenate([self.response, other.response])
        self.cell_line_ids = np.concatenate([self.cell_line_ids, other.cell_line_ids])
        self.drug_ids = np.concatenate([self.drug_ids, other.drug_ids])

        if self.predictions is not None and other.predictions is not None:
            self.predictions = np.concatenate([self.predictions, other.predictions])

    def remove_nan_responses(self) -> None:
        """
        Removes rows with NaN values in the response
        """
        mask = np.isnan(self.response)
        self.response = self.response[~mask]
        self.cell_line_ids = self.cell_line_ids[~mask]
        self.drug_ids = self.drug_ids[~mask]
        if self.predictions is not None:
            self.predictions = self.predictions[~mask]

    def shuffle(self, random_state: int = 42) -> None:
        """
        Shuffles the dataset.
        :param random_state: random state
        """
        indices = np.arange(len(self.response))
        np.random.seed(random_state)
        np.random.shuffle(indices)
        self.response = self.response[indices]
        self.cell_line_ids = self.cell_line_ids[indices]
        self.drug_ids = self.drug_ids[indices]
        if self.predictions is not None:
            self.predictions = self.predictions[indices]

    def remove_drugs(self, drugs_to_remove: Union[str, list]) -> None:
        """
        Removes drugs from the dataset.
        :param drugs_to_remove: name of drug or list of names of multiple drugs to remove
        """
        if isinstance(drugs_to_remove, str):
            drugs_to_remove = [drugs_to_remove]

        mask = [drug not in drugs_to_remove for drug in self.drug_ids]
        self.drug_ids = self.drug_ids[mask]
        self.cell_line_ids = self.cell_line_ids[mask]
        self.response = self.response[mask]

    def remove_cell_lines(self, cell_lines_to_remove: Union[str, list]) -> None:
        """
        Removes cell lines from the dataset.
        :param cell_lines_to_remove: name of cell line or list of names of multiple cell lines to
        remove
        """
        if isinstance(cell_lines_to_remove, str):
            cell_lines_to_remove = [cell_lines_to_remove]

        mask = [
            cell_line not in cell_lines_to_remove for cell_line in self.cell_line_ids
        ]
        self.drug_ids = self.drug_ids[mask]
        self.cell_line_ids = self.cell_line_ids[mask]
        self.response = self.response[mask]

    def remove_rows(self, indices: ArrayLike) -> None:
        """
        Removes rows from the dataset.
        :param indices: indices of rows to remove
        """
        self.drug_ids = np.delete(self.drug_ids, indices)
        self.cell_line_ids = np.delete(self.cell_line_ids, indices)
        self.response = np.delete(self.response, indices)
        if self.predictions is not None:
            self.predictions = np.delete(self.predictions, indices)

    def reduce_to(
        self, cell_line_ids: Optional[ArrayLike], drug_ids: Optional[ArrayLike]
    ) -> None:
        """
        Removes all rows which contain a cell_line not in cell_line_ids or a drug not in drug_ids
        :param cell_line_ids: cell line IDs or None to keep all cell lines
        :param drug_ids: drug IDs or None to keep all cell lines
        """
        if drug_ids is not None:
            self.remove_drugs(list(set(self.drug_ids) - set(drug_ids)))

        if cell_line_ids is not None:
            self.remove_cell_lines(list(set(self.cell_line_ids) - set(cell_line_ids)))

    def split_dataset(
        self,
        n_cv_splits,
        mode,
        split_validation=True,
        split_early_stopping=True,
        validation_ratio=0.1,
        random_state=42,
    ) -> List[dict]:
        """
        Splits the dataset into training, validation and test sets for cross-validation
        :param n_cv_splits: number of cross-validation splits, e.g., 5
        :param mode: split mode ('LPO', 'LCO', 'LDO')
        :param split_validation: if True, a validation set is generated
        :param split_early_stopping: if True, an early stopping set is generated
        :param validation_ratio: ratio of validation set size to training set size
        :param random_state: random state
        :return: list of dictionaries containing the cross-validation datasets. Each fold is a
        dictionary with keys 'train', 'validation', 'test', 'validation_es', 'early_stopping'.
        """
        cell_line_ids = self.cell_line_ids
        drug_ids = self.drug_ids
        response = self.response

        if mode == "LPO":
            cv_splits = leave_pair_out_cv(
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
            cv_splits = leave_group_out_cv(
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
            raise ValueError(
                f"Unknown split mode '{mode}'. Choose from 'LPO', 'LCO', 'LDO'."
            )

        if split_validation and split_early_stopping:
            for split in cv_splits:
                validation_es, early_stopping = split_early_stopping_data(
                    split["validation"], test_mode=mode
                )
                split["validation_es"] = validation_es
                split["early_stopping"] = early_stopping
        self.cv_splits = cv_splits
        return cv_splits

    def save_splits(self, path: str) -> None:
        """
        Save cross validation splits to
        path/cv_split_0_train.csv
        path/cv_split_0_test.csv
        :param path: path to the directory where the cv split files are saved
        """
        assert (
            self.cv_splits is not None
        ), "trying to save splits, but DrugResponseDataset was not split."
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
        Load cross validation splits from
        path/cv_split_0_train.csv
        path/cv_split_0_test.csv
        :param path: path to the directory containing the cv split files
        """
        files = os.listdir(path)
        files = [
            file
            for file in files
            if (file.endswith(".csv") and file.startswith("cv_split"))
        ]
        assert len(files) > 0, f"No cv split files found in {path}"

        train_splits = [file for file in files if "train" in file]
        test_splits = [file for file in files if "test" in file]

        validation_es_splits = [file for file in files if "validation_es" in file]
        validation_splits = [
            file
            for file in files
            if "validation" in file and file not in validation_es_splits
        ]
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
        self.cv_splits = []

        for split_train, split_test in zip(train_splits, test_splits):
            tr_split = DrugResponseDataset(dataset_name=self.dataset_name)
            tr_split.load(os.path.join(path, split_train))

            te_split = DrugResponseDataset(dataset_name=self.dataset_name)
            te_split.load(os.path.join(path, split_test))
            self.cv_splits.append({"train": tr_split, "test": te_split})

        for mode in ["validation", "validation_es", "early_stopping"]:
            if len(optional_splits[mode]) > 0:
                for i, v_split in enumerate(optional_splits[mode]):
                    split = DrugResponseDataset(dataset_name=self.dataset_name)
                    split.load(os.path.join(path, v_split))
                    self.cv_splits[i][mode] = split

    def copy(self):
        """
        Returns a copy of the drug response dataset.
        """
        return DrugResponseDataset(
            response=copy.deepcopy(self.response),
            cell_line_ids=copy.deepcopy(self.cell_line_ids),
            drug_ids=copy.deepcopy(self.drug_ids),
            predictions=copy.deepcopy(self.predictions),
            dataset_name=self.dataset_name,
        )

    def __hash__(self):
        return hash(
            (
                self.dataset_name,
                tuple(self.cell_line_ids),
                tuple(self.drug_ids),
                tuple(self.response),
                tuple(self.predictions) if self.predictions is not None else None,
            )
        )

    def mask(self, mask: List[bool]) -> None:
        """
        Masks the dataset.
        :param mask: boolean mask
        """
        self.response = self.response[mask]
        self.cell_line_ids = self.cell_line_ids[mask]
        self.drug_ids = self.drug_ids[mask]
        if self.predictions is not None:
            self.predictions = self.predictions[mask]

    def transform(self, response_transformation: TransformerMixin) -> None:
        """
        Apply transformation to the response data and prediction data of the dataset.
        :param response_transformation: e.g., StandardScaler, MinMaxScaler, RobustScaler
        """
        self.response = response_transformation.transform(
            self.response.reshape(-1, 1)
        ).squeeze()
        if self.predictions is not None:
            self.predictions = response_transformation.transform(
                self.predictions.reshape(-1, 1)
            ).squeeze()

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
        self.response = response_transformation.inverse_transform(
            self.response.reshape(-1, 1)
        ).squeeze()
        if self.predictions is not None:
            self.predictions = response_transformation.inverse_transform(
                self.predictions.reshape(-1, 1)
            ).squeeze()


def split_early_stopping_data(
    validation_dataset: DrugResponseDataset, test_mode: str
) -> Tuple[DrugResponseDataset, DrugResponseDataset]:
    """
    Splits the validation dataset into a validation and an early stopping dataset.
    :param validation_dataset: input validation dataset
    :param test_mode: LCO, LDO, LPO
    :return: the resulting validation and early stopping datasets
    """
    validation_dataset.shuffle(random_state=42)
    cv_v = validation_dataset.split_dataset(
        n_cv_splits=4,
        mode=test_mode,
        split_validation=False,
        split_early_stopping=False,
        random_state=42,
    )
    # take the first fold of a 4 cv as the split i.e., 3/4 for validation and 1/4 for early stopping
    validation_dataset = cv_v[0]["train"]
    early_stopping_dataset = cv_v[0]["test"]
    return validation_dataset, early_stopping_dataset


def leave_pair_out_cv(
    n_cv_splits: int,
    response: ArrayLike,
    cell_line_ids: ArrayLike,
    drug_ids: ArrayLike,
    split_validation=True,
    validation_ratio=0.1,
    random_state=42,
    dataset_name: Optional[str] = None,
) -> List[dict]:
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
    :return: list of dicts of the cross validation sets
    """
    assert (
        len(response) == len(cell_line_ids) == len(drug_ids)
    ), "response, cell_line_ids and drug_ids must have the same length"
    indices = np.arange(len(response))
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(indices)
    response = response[shuffled_indices].copy()
    cell_line_ids = cell_line_ids[shuffled_indices].copy()
    drug_ids = drug_ids[shuffled_indices].copy()

    # We use GroupKFold to ensure that each pair is only in one fold (prevent data leakage due to
    # experimental replicates).
    # If there are no replicates this is equivalent to KFold.
    groups = [cell + "_" + drug for cell, drug in zip(cell_line_ids, drug_ids)]
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


def leave_group_out_cv(
    group: str,
    n_cv_splits: int,
    response: ArrayLike,
    cell_line_ids: ArrayLike,
    drug_ids: ArrayLike,
    split_validation=True,
    validation_ratio=0.1,
    random_state=42,
    dataset_name: Optional[str] = None,
):
    """
    Leave group out cross validation: Splits data into n_cv_splits number of cross validation
    splits.
    :param group: group to leave out (cell_line or drug)
    :param n_cv_splits: number of cross validation splits
    :param random_state: random state
    :return: list of dicts of the cross validation sets
    """

    assert group in {
        "cell_line",
        "drug",
    }, f"group must be 'cell_line' or 'drug', but is {group}"

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
            # split training set into training and validation set. The validation set also does
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
    """
    Class for feature datasets.
    """

    def __init__(
        self,
        features: Dict[str, Dict[str, Any]],
        meta_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the feature dataset.
        :param features: dictionary of features,
            key: drug ID/cell line ID, value: Dict of feature views,
            key: feature name, value: feature vector
        :param meta_info: additional information for the views, e.g., gene names for gene expression
        """
        super().__init__()
        self.features = features
        self.view_names = self.get_view_names()
        if meta_info is not None:
            # assert that str of meta Dict[str, Any] is in view_names
            assert all(
                meta_key in self.view_names for meta_key in meta_info.keys()
            ), f"Meta keys {meta_info.keys()} not in view names {self.view_names}"
            self.meta_info = meta_info
        else:
            self.meta_info = None
        self.identifiers = self.get_ids()

    def save(self, path: str):
        """
        Saves the feature dataset to data.
        :param path: path to the dataset
        """
        raise NotImplementedError("save method not implemented")

    def load(self, path: str):
        """
        Loads the feature dataset from data.
        :param path: path to the dataset
        """
        raise NotImplementedError("load method not implemented")

    def randomize_features(
        self, views_to_randomize: Union[str, list], randomization_type: str
    ) -> None:
        """
        Randomizes the feature vectors.
        :param views_to_randomize: name of feature view or list of names of multiple feature views
            to randomize. The other views are not randomized.
        :param randomization_type: randomization type ('permutation', 'invariant').
        :return:
        Permutation permutes the feature vectors.
        Invariant means that the randomization is done in a way that a key characteristic of the
        feature is preserved. In case of matrices, this is the mean and standard deviation of the
        feature view for this instance, for networks it is the degree distribution.
        """
        assert randomization_type in [
            "permutation",
            "invariant",
        ], (
            f"Unknown randomization type '{randomization_type}'. "
            f"Choose from 'permutation', 'invariant'."
        )

        if isinstance(views_to_randomize, str):
            views_to_randomize = [views_to_randomize]

        if randomization_type == "permutation":
            # Permute the specified views for each entity (= cell line or drug)
            # E.g. each cell line gets the feature vector/graph/image... of another cell line.
            # Drawn without replacement.
            self.features = permute_features(
                features=self.features,
                views_to_permute=views_to_randomize,
                identifiers=self.identifiers,
                all_views=self.view_names,
            )

        elif randomization_type == "invariant":
            # Invariant randomization: Randomize the specified views for each entity in a way that
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
                    elif isinstance(
                        self.features[identifier][view], nx.classes.graph.Graph
                    ):
                        new_features = randomize_graph(self.features[identifier][view])

                    else:
                        raise ValueError(
                            f"No invariant randomization available for feature view "
                            f"type '{type(self.features[identifier][view])}'."
                        )
                    self.features[identifier][view] = new_features

    def get_ids(self):
        """
        returns drug ids of the dataset
        """
        return np.array(list(self.features.keys()))

    def get_view_names(self):
        """
        returns feature view names
        """
        return list(self.features[list(self.features.keys())[0]].keys())

    def get_feature_matrix(
        self, view: str, identifiers: ArrayLike, stack: bool = True
    ) -> Union[np.ndarray, List]:
        """
        Returns the feature matrix for the given view. The feature view must be a vector or matrix.
        :param view: view name
        :param identifiers: list of identifiers (cell lines oder drugs)
        :param stack: if True, stacks the feature vectors to a matrix.
            If False, returns a list of features.
        :return: feature matrix
        """
        assert len(identifiers) > 0, "get_feature_matrix: No identifiers given."

        assert view in self.view_names, f"View '{view}' not in in the FeatureDataset."
        missing_identifiers = {
            id_ for id_ in identifiers if id_ not in self.identifiers
        }
        assert not missing_identifiers, (
            f"{len(missing_identifiers)} of {len(np.unique(identifiers))} ids are not in the "
            f"FeatureDataset. Missing ids: {missing_identifiers}"
        )

        assert all(
            len(self.features[id_][view]) == len(self.features[identifiers[0]][view])
            for id_ in identifiers
        ), f"Feature vectors of view {view} have different lengths."

        assert all(
            isinstance(self.features[id_][view], np.ndarray) for id_ in identifiers
        ), f"get_feature_matrix only works for vectors or matrices. {view} is not a numpy array."
        out = [self.features[id_][view] for id_ in identifiers]
        return np.stack(out, axis=0) if stack else out

    def copy(self):
        """
        Returns a copy of the feature dataset.
        """
        return FeatureDataset(features=copy.deepcopy(self.features))

    def add_features(self, other: "FeatureDataset") -> None:
        """
        Adds features views from another dataset. Inner join (only common identifiers are kept).
        :param other: other dataset
        """
        assert (
            len(set(self.view_names) & set(other.view_names)) == 0
        ), "Trying to add features but feature views overlap. FeatureDatasets should be distinct."
        if other.meta_info is not None:
            self.add_meta_info(other)

        common_identifiers = set(self.identifiers).intersection(other.identifiers)
        new_features = {}
        for id_ in common_identifiers:
            new_features[id_] = {
                view: self.features[id_][view] for view in self.view_names
            }
            for view in other.view_names:
                new_features[id_][view] = other.features[id_][view]

        self.features = new_features
        self.view_names = self.get_view_names()
        self.identifiers = self.get_ids()

    def add_meta_info(self, other: "FeatureDataset") -> None:
        """
        Adds meta information to the feature dataset.
        :param other: other dataset
        """
        other_meta = other.meta_info
        self.meta_info.update(other_meta)

    def transform_features(
        self, ids: ArrayLike, transformer: TransformerMixin, view: str
    ):
        """
        Applies a transformation like standard scaling to features.
        :param ids: The IDs to transform
        :param transformer: fitted sklearn transformer
        :param view: the view to transform
        """
        assert (
            view in self.view_names
        ), f"Transform view '{view}' not in in the FeatureDataset."
        assert all(
            [clid in self.features for clid in ids]
        ), "Trying to transform, but a cell line is missing."

        assert len(np.unique(ids)) == len(ids), f"IDs should be unique."

        for identifier in ids:
            feature_vector = self.features[identifier][view]
            scaled_feature_vector = transformer.transform([feature_vector])[0]
            self.features[identifier][view] = scaled_feature_vector

    def fit_transform_features(
        self, train_ids: ArrayLike, transformer: TransformerMixin, view: str
    ):
        """
        Fits and applies a transformation. Fitting is done only on the train_ids.
        :param train_ids: The IDs corresponding to the training dataset.
        :param transformer: sklearn transformer
        :param view: the view to transform
        :return: The modified FeatureDataset with transformed gene expression features.
        """
        assert (
            view in self.view_names
        ), f"Transform view '{view}' not in in the FeatureDataset."

        assert len(np.unique(train_ids)) == len(
            train_ids
        ), f"Train IDs should be unique."

        train_features = []

        # Collect all features of the view for fitting the scaler
        for identifier in train_ids:
            feature_vector = self.features[identifier][view]
            train_features.append(feature_vector)

        # Fit the scaler on the collected feature data
        train_features = np.vstack(train_features)
        transformer.fit(train_features)

        # Apply transformation and scaling to each feature vector
        for identifier in self.features:
            feature_vector = self.features[identifier][view]
            scaled_gene_expression = transformer.transform([feature_vector])[0]
            self.features[identifier][view] = scaled_gene_expression
        return transformer

    def apply(self, function: Callable, view: str):
        """
        Applies a function to the features of a view.
        """
        for identifier in self.features:
            self.features[identifier][view] = function(self.features[identifier][view])
