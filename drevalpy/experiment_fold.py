"""Shared CV-fold preparation for experiment and CLI paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models._model_lookup import is_single_drug_model_name
from drevalpy.models.drp_model import DRPModel
from drevalpy.pipeline_function import pipeline_function


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
) -> tuple[DrugResponseDataset, DrugResponseDataset, DrugResponseDataset | None, DrugResponseDataset]:
    """
    Get train, validation, (early stopping), and test datasets from the CV split.

    Returns copies of the datasets to prevent in-place modifications (e.g., add_rows,
    reduce_to) from affecting the original split data used by subsequent models.

    :param split: CV split dictionary with train/validation/test (+ optional ES) keys
    :param model_class: model class used to decide early-stopping partitions
    :param model_name: model name used for single-drug masking
    :param drug_id: drug identifier for single-drug models
    :returns: train, validation, early_stopping (or None), and test datasets
    """
    fold = prepare_fold_datasets(split, model_class, model_name, drug_id)
    return fold.train, fold.validation, fold.early_stopping, fold.test


def prepare_fold_datasets(
    split: dict[str, DrugResponseDataset],
    model_class: type[DRPModel],
    model_name: str,
    drug_id: str | None = None,
) -> FoldDatasets:
    """
    Copy/mask fold partitions with the same logic for experiment and CLI callers.

    :param split: CV split dictionary with train/validation/test (+ optional ES) keys
    :param model_class: model class used to decide early-stopping partitions
    :param model_name: model name used for single-drug masking
    :param drug_id: drug identifier for single-drug models
    :returns: fold datasets with copies and optional drug masks applied
    """
    train_dataset = split["train"].copy()
    validation_dataset = split["validation"].copy()
    test_dataset = split["test"].copy()

    if model_class.early_stopping:
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
    """
    Return a shuffled train+validation copy for final fold training.

    :param train_dataset: training dataset
    :param validation_dataset: validation dataset to merge into training
    :param random_state: shuffle seed
    :returns: shuffled concatenated train+validation dataset
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
    """
    Prepare fold data with train+validation already merged for final training.

    :param split: CV split dictionary with train/validation/test (+ optional ES) keys
    :param model_class: model class used to decide early-stopping partitions
    :param model_name: model name used for single-drug masking
    :param drug_id: drug identifier for single-drug models
    :param random_state: shuffle seed for the merged train set
    :returns: fold datasets with merged train and optional early stopping
    """
    fold = prepare_fold_datasets(split, model_class, model_name, drug_id)
    merged_train = merge_train_validation(fold.train, fold.validation, random_state=random_state)
    return FoldDatasets(
        train=merged_train,
        validation=fold.validation,
        early_stopping=fold.early_stopping if model_class.early_stopping else None,
        test=fold.test,
    )


def early_stopping_for_model(
    model_or_class: Any,
    early_stopping_dataset: DrugResponseDataset | None,
) -> DrugResponseDataset | None:
    """
    Return early-stopping data only when the model supports it.

    :param model_or_class: model instance or class with an ``early_stopping`` attribute
    :param early_stopping_dataset: candidate early-stopping dataset
    :returns: ``early_stopping_dataset`` when supported, otherwise ``None``
    """
    supports = bool(getattr(model_or_class, "early_stopping", False))
    return early_stopping_dataset if supports else None
