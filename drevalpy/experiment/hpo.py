"""Hyperparameter selection helpers for experiment workflows."""

from __future__ import annotations

import os
from typing import Any

from sklearn.base import TransformerMixin

from drevalpy.components.tuning.hpo import hpam_tune

from ..datasets.dataset import DrugResponseDataset
from ..models.drp_model import DRPModel


def select_fold_hyperparameters(
    *,
    model_class: type[DRPModel],
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    response_transformation: TransformerMixin | None,
    metric: str,
    path_data: str,
    model_checkpoint_dir: str,
    hyperparameter_tuning: bool,
    hpo_config: Any,
    wandb_project: str | None = None,
    split_index: int | None = None,
    wandb_base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Resolve hyperparameters for one CV fold (tuning or defaults).

    :returns: flat hyperparameter mapping for ``model_class(...)``
    """
    from drevalpy.components.tuning.drp_hyperparameters import (
        has_tunable_hyperparameters,
    )

    if not hyperparameter_tuning or not has_tunable_hyperparameters(model_class):
        return model_class.get_default_hyperparameters()

    tuning_inputs: dict[str, Any] = {
        "model_class": model_class,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "early_stopping_dataset": early_stopping_dataset,
        "response_transformation": response_transformation,
        "metric": metric,
        "path_data": path_data,
        "model_checkpoint_dir": model_checkpoint_dir,
        "hpo_config": hpo_config,
    }
    if wandb_project is not None:
        tuning_inputs["wandb_project"] = wandb_project
        tuning_inputs["split_index"] = split_index
        tuning_inputs["wandb_base_config"] = wandb_base_config
    return hpam_tune(**tuning_inputs)


def select_final_model_hyperparameters(
    *,
    model_class: type[DRPModel],
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    response_transformation: TransformerMixin | None,
    metric: str,
    path_data: str,
    model_checkpoint_dir: str,
    hyperparameter_tuning: bool,
    hpo_config: Any,
) -> dict[str, Any]:
    """Resolve hyperparameters for final full-data model training."""
    from drevalpy.components.tuning.drp_hyperparameters import (
        has_tunable_hyperparameters,
    )

    default_hpams = model_class.get_default_hyperparameters()
    if not hyperparameter_tuning or not has_tunable_hyperparameters(model_class):
        return default_hpams
    return hpam_tune(
        model_class=model_class,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        early_stopping_dataset=early_stopping_dataset,
        response_transformation=response_transformation,
        metric=metric,
        path_data=path_data,
        model_checkpoint_dir=model_checkpoint_dir,
        hpo_config=hpo_config,
    )


def fold_hpo_storage_path(result_path: str) -> str:
    """Absolute Ray Tune storage directory for nested-CV HPO."""
    return os.path.abspath(os.path.join(result_path, "raytune"))


def final_model_hpo_storage_path(result_path: str) -> str:
    """Absolute Ray Tune storage directory for final-model HPO."""
    return os.path.abspath(os.path.join(result_path, "raytune_final"))
