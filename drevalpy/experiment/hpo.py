"""Hyperparameter selection helpers for experiment workflows."""

from __future__ import annotations

from typing import Any

from sklearn.base import TransformerMixin
from upath import UPath as Path

from ..data.mudataset import MuDataset
from ..data.splitting import EntityScope
from ..models.drp_model import DRPModel


def select_fold_hyperparameters(
    *,
    model_class: type[DRPModel],
    mudataset: MuDataset,
    train_scope: EntityScope,
    val_scope: EntityScope,
    early_stopping_scope: EntityScope | None,
    response_transformation: TransformerMixin | None,
    metric: str,
    model_checkpoint_dir: str | Path | None,
    hyperparameter_tuning: bool,
    hpo_config: Any,
    wandb_project: str | None = None,
    split_index: int | None = None,
    wandb_base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve hyperparameters for one CV fold (tuning or defaults).

    :param model_class: Model class to tune or instantiate with defaults.
    :param mudataset: Full dataset shared across all folds.
    :param train_scope: Training EntityScope for the fold.
    :param val_scope: Validation EntityScope for scoring.
    :param early_stopping_scope: Optional early-stopping scope.
    :param response_transformation: Optional response transformer.
    :param metric: Metric optimized during HPO.
    :param model_checkpoint_dir: Directory for model checkpoints, or ``None`` for a temporary one.
    :param hyperparameter_tuning: Whether to run HPO when tunable parameters exist.
    :param hpo_config: Ray Tune / Optuna configuration object.
    :param wandb_project: Optional Weights & Biases project name.
    :param split_index: CV fold index for W&B logging.
    :param wandb_base_config: Base W&B config merged per trial.

    :returns: Flat hyperparameter mapping for ``model_class(...)``.
    """
    from drevalpy.components.tuning.drp_hyperparameters import (
        has_tunable_hyperparameters,
    )
    from drevalpy.components.tuning.hpo import mu_hpam_tune

    if not hyperparameter_tuning or not has_tunable_hyperparameters(model_class):
        return model_class.get_default_hyperparameters()

    tuning_inputs: dict[str, Any] = {
        "model_class": model_class,
        "mudataset": mudataset,
        "train_scope": train_scope,
        "val_scope": val_scope,
        "early_stopping_scope": early_stopping_scope,
        "response_transformation": response_transformation,
        "metric": metric,
        "model_checkpoint_dir": model_checkpoint_dir,
        "hpo_config": hpo_config,
    }
    if wandb_project is not None:
        tuning_inputs["wandb_project"] = wandb_project
        tuning_inputs["split_index"] = split_index
        tuning_inputs["wandb_base_config"] = wandb_base_config
    return mu_hpam_tune(**tuning_inputs)


def select_final_model_hyperparameters(
    *,
    model_class: type[DRPModel],
    mudataset: MuDataset,
    train_scope: EntityScope,
    val_scope: EntityScope,
    early_stopping_scope: EntityScope | None,
    response_transformation: TransformerMixin | None,
    metric: str,
    model_checkpoint_dir: str | Path | None,
    hyperparameter_tuning: bool,
    hpo_config: Any,
) -> dict[str, Any]:
    """Resolve hyperparameters for final full-data model training.

    :param model_class: Model class to tune or instantiate with defaults.
    :param mudataset: Full dataset shared across all folds.
    :param train_scope: Training EntityScope for the holdout.
    :param val_scope: Validation EntityScope for scoring.
    :param early_stopping_scope: Optional early-stopping scope.
    :param response_transformation: Optional response transformer.
    :param metric: Metric optimized during HPO.
    :param model_checkpoint_dir: Directory for model checkpoints, or ``None`` for a temporary one.
    :param hyperparameter_tuning: Whether to run HPO when tunable parameters exist.
    :param hpo_config: Ray Tune / Optuna configuration object.

    :returns: Flat hyperparameter mapping for ``model_class(...)``.
    """
    from drevalpy.components.tuning.drp_hyperparameters import (
        has_tunable_hyperparameters,
    )
    from drevalpy.components.tuning.hpo import mu_hpam_tune

    default_hpams = model_class.get_default_hyperparameters()
    if not hyperparameter_tuning or not has_tunable_hyperparameters(model_class):
        return default_hpams
    return mu_hpam_tune(
        model_class=model_class,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=early_stopping_scope,
        response_transformation=response_transformation,
        metric=metric,
        model_checkpoint_dir=model_checkpoint_dir,
        hpo_config=hpo_config,
    )


def fold_hpo_storage_path(result_path: str | Path) -> str:
    """Absolute Ray Tune storage directory for nested-CV HPO.

    Returns a ``str``: Ray also accepts ``s3://`` / ``gs://`` URIs for
    ``storage_path``, and ``Path`` would collapse ``s3://b/x`` to ``s3:/b/x``.

    :param result_path: Experiment result root directory.

    :returns: Absolute path to the nested-CV Ray Tune storage folder.
    """
    return str((Path(result_path) / "raytune").absolute())


def final_model_hpo_storage_path(result_path: str | Path) -> str:
    """Absolute Ray Tune storage directory for final-model HPO.

    Returns a ``str`` for the same URI-safety reason as ``fold_hpo_storage_path``.

    :param result_path: Experiment result root directory.

    :returns: Absolute path to the final-model Ray Tune storage folder.
    """
    return str((Path(result_path) / "raytune_final").absolute())
