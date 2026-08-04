"""Shared CV-fold preparation for experiment and CLI paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models._model_lookup import is_single_drug_model_name
from drevalpy.models.drp_model import DRPModel
from drevalpy.utils._pipeline_function import pipeline_function


@dataclass(frozen=True)
class FoldDatasets:
    """Copied and optionally drug-masked datasets for one CV fold."""

    train: DrugResponseDataset
    validation: DrugResponseDataset
    early_stopping: DrugResponseDataset | None
    test: DrugResponseDataset


@pipeline_function
def get_datasets_from_cv_split(
    split: dict[str, DrugResponseDataset],
    model_class: type[DRPModel],
    model_name: str,
    drug_id: str | None = None,
) -> tuple[
    DrugResponseDataset,
    DrugResponseDataset,
    DrugResponseDataset | None,
    DrugResponseDataset,
]:
    """Extract train, validation, early-stopping, and test sets from a CV fold.

    Returns copies so in-place edits do not affect other models on the same fold.

      Args:
          split: CV split dict with ``train``, ``validation``, and ``test`` keys.
          model_class: Model class used to decide early-stopping partitions.
          model_name: Run key used for single-drug masking.
          drug_id: Drug identifier for single-drug models.

      Returns:
          Train, validation, early-stopping (or ``None``), and test datasets.
    """
    fold = prepare_fold_datasets(split, model_class, model_name, drug_id)
    return fold.train, fold.validation, fold.early_stopping, fold.test


def prepare_fold_datasets(
    split: dict[str, DrugResponseDataset],
    model_class: type[DRPModel],
    model_name: str,
    drug_id: str | None = None,
) -> FoldDatasets:
    """Copy and optionally drug-mask fold partitions for experiment runners.

    Args:
        split: CV split dict with ``train``, ``validation``, and ``test`` keys.
        model_class: Model class used to decide early-stopping partitions.
        model_name: Run key used for single-drug masking.
        drug_id: Drug identifier for single-drug models.

    Returns:
        Fold datasets with copies and optional drug masks applied.
    """
    train_dataset = split["train"].copy()
    validation_dataset = split["validation"].copy()
    test_dataset = split["test"].copy()

    if model_class.supports_early_stopping():
        validation_dataset = split["validation_es"].copy()
        early_stopping_dataset = split["early_stopping"].copy()
    else:
        early_stopping_dataset = None

    if is_single_drug_model_name(model_name):
        train_dataset = train_dataset.masked(train_dataset.drug_ids == drug_id)
        validation_dataset = validation_dataset.masked(validation_dataset.drug_ids == drug_id)
        test_dataset = test_dataset.masked(test_dataset.drug_ids == drug_id)
        if early_stopping_dataset is not None:
            early_stopping_dataset = early_stopping_dataset.masked(early_stopping_dataset.drug_ids == drug_id)

    return FoldDatasets(
        train=train_dataset,
        validation=validation_dataset,
        early_stopping=early_stopping_dataset,
        test=test_dataset,
    )


def merge_train_validation(
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    *,
    random_state: int = 42,
) -> DrugResponseDataset:
    """Return a shuffled train-plus-validation copy for final fold training.

    Args:
        train_dataset: Training dataset.
        validation_dataset: Validation dataset to merge into training.
        random_state: Shuffle seed.

    Returns:
        Shuffled concatenation of *train_dataset* and *validation_dataset*.
    """
    return train_dataset.with_rows_added(validation_dataset).shuffled(random_state=random_state)


def prepare_final_fold_training_data(
    split: dict[str, DrugResponseDataset],
    model_class: type[DRPModel],
    model_name: str,
    drug_id: str | None = None,
    *,
    random_state: int = 42,
) -> FoldDatasets:
    """Prepare fold data with train and validation merged for final training.

    Args:
        split: CV split dict with ``train``, ``validation``, and ``test`` keys.
        model_class: Model class used to decide early-stopping partitions.
        model_name: Run key used for single-drug masking.
        drug_id: Drug identifier for single-drug models.
        random_state: Shuffle seed for the merged train set.

    Returns:
        Fold datasets with merged train and optional early stopping.
    """
    fold = prepare_fold_datasets(split, model_class, model_name, drug_id)
    merged_train = merge_train_validation(fold.train, fold.validation, random_state=random_state)
    return FoldDatasets(
        train=merged_train,
        validation=fold.validation,
        early_stopping=fold.early_stopping if model_class.supports_early_stopping() else None,
        test=fold.test,
    )


def early_stopping_for_model(
    model_or_class: Any,
    early_stopping_dataset: DrugResponseDataset | None,
) -> DrugResponseDataset | None:
    """Return early-stopping data only when the model supports it.

    Args:
        model_or_class: Model instance or class with early-stopping capability.
        early_stopping_dataset: Candidate early-stopping dataset.

    Returns:
        *early_stopping_dataset* when supported, otherwise ``None``.
    """
    supports_fn = getattr(model_or_class, "supports_early_stopping", None)
    if callable(supports_fn):
        supports = bool(supports_fn())
    else:
        supports = bool(getattr(model_or_class, "early_stopping", False))
    return early_stopping_dataset if supports else None


def make_train_val_split_impl(
    dataset: DrugResponseDataset,
    test_mode: str,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[DrugResponseDataset, DrugResponseDataset]:
    """
    Split a dataset into train and validation sets according to the test mode and desired ratio.

    :param dataset: full dataset to split
    :param test_mode: one of "LPO", "LCO", "LDO", "LTO"
    :param val_ratio: approximate fraction of data to use for validation
    :param random_state: random seed
    :returns: (train_dataset, validation_dataset)
    :raises ValueError: if no tissue information is provided for the DrugResponseDataset
    """
    if test_mode == "LTO":
        if dataset.tissue is not None:
            n_groups = len(np.unique(dataset.tissue))
        else:
            raise ValueError("Tissue information is missing but required for LTO mode.")
    elif test_mode == "LCO":
        n_groups = len(np.unique(dataset.cell_line_ids))
    elif test_mode == "LDO":
        n_groups = len(np.unique(dataset.drug_ids))
    else:
        n_groups = len(dataset)

    n_splits = int(1 / val_ratio)
    n_splits = min(n_splits, n_groups)

    split = dataset.split_dataset(
        n_cv_splits=n_splits,
        mode=test_mode,
        split_validation=False,
        random_state=random_state,
    )[0]

    return split["train"], split["test"]
