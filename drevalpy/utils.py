import os
import shutil
from typing import List, Optional
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr, kendalltau
from pingouin import partial_corr

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
    :return: list of dicts of the cross validation sets
    """

    from drevalpy.datasets.dataset import DrugResponseDataset

    assert (
        len(response) == len(cell_line_ids) == len(drug_ids)
    ), "response, cell_line_ids and drug_ids must have the same length"

    kf = KFold(n_splits=n_cv_splits, shuffle=True, random_state=random_state)
    cv_sets = []

    for train_indices, test_indices in kf.split(response):
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
    Leave group out cross validation. Splits data into n_cv_splits number of cross validation splits.
    :param group: group to leave out (cell_line or drug)
    :param n_cv_splits: number of cross validation splits
    :param random_state: random state
    :return: list of dicts of the cross validation sets
    """
    from drevalpy.datasets.dataset import DrugResponseDataset

    assert group in {
        "cell_line",
        "drug",
    }, f"group must be 'cell_line' or 'drug', but is {group}"

    if group == "cell_line":
        group_ids = cell_line_ids
    elif group == "drug":
        group_ids = drug_ids

    # shuffle, since GroupKFold does not implement this
    indices = np.arange(len(response))
    shuffled_indices = np.random.RandomState(seed=random_state).permutation(indices)
    response = response[shuffled_indices]
    cell_line_ids = cell_line_ids[shuffled_indices]
    drug_ids = drug_ids[shuffled_indices]
    group_ids = group_ids[shuffled_indices]

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
            # split training set into training and validation set. The validation set also does contain unqiue cell lines/drugs
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


def partial_correlation(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    cell_line_ids: np.ndarray,
    drug_ids: np.ndarray,
    method: str = "pearson",
    return_pvalue: bool = False,

) -> float:
    """
    Computes the partial correlation between predictions and response, conditioned on cell line and drug.
    :param y_pred: predictions
    :param y_true: response
    :param cell_line_ids: cell line IDs
    :param drug_ids: drug IDs
    :param method: method to compute the partial correlation (pearson, spearman)
    :return: partial correlation float
    """

    assert (
        len(y_pred) == len(y_true) == len(cell_line_ids) == len(drug_ids)
    ), "predictions, response, drug_ids, and cell_line_ids must have the same length"

    df = pd.DataFrame(
        {
            "response": y_true,
            "predictions": y_pred,
            "cell_line_ids": cell_line_ids,
            "drug_ids": drug_ids,
        }
    )

    df["cell_line_ids"] = pd.factorize(df["cell_line_ids"])[0]
    df["drug_ids"] = pd.factorize(df["drug_ids"])[0]
    result = partial_corr(data=df, x='predictions', y='response', covar=['cell_line_ids', 'drug_ids'], method=method)
    r = result['r'].iloc[0]
    p = result['p-val'].iloc[0]
    if return_pvalue:
        return r, p
    else:
        return r


def pearson(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes the pearson correlation between predictions and response.
    :param y_pred: predictions
    :param y_true: response
    :return: pearson correlation float
    """

    assert len(y_pred) == len(
        y_true
    ), "predictions, response  must have the same length"
    if (y_pred == y_pred[0]).all() or (y_true == y_true[0]).all() or len(y_true) < 2:
        return np.nan
    return pearsonr(y_pred, y_true)[0]


def spearman(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes the spearman correlation between predictions and response.
    :param y_pred: predictions
    :param y_true: response
    :return: spearman correlation float
    """
    # we can use scipy.stats.spearmanr
    assert len(y_pred) == len(
        y_true
    ), "predictions, response  must have the same length"
    if (y_pred == y_pred[0]).all() or (y_true == y_true[0]).all() or len(y_true) < 2:
        return np.nan
    return spearmanr(y_pred, y_true)[0]


def kendall(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes the kendall tau correlation between predictions and response.
    :param y_pred: predictions
    :param y_true: response
    :return: kendall tau correlation float
    """
    # we can use scipy.stats.spearmanr
    assert len(y_pred) == len(
        y_true
    ), "predictions, response  must have the same length"
    if (y_pred == y_pred[0]).all() or (y_true == y_true[0]).all() or len(y_true) < 2:
        return np.nan
    return kendalltau(y_pred, y_true)[0]


def handle_overwrite(path: str, overwrite: bool) -> None:
    """Handle overwrite logic for a given path."""
    if os.path.exists(path) and overwrite:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
