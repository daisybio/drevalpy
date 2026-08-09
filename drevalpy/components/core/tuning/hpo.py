"""Optuna hyperparameter optimization for DRPModel experiments."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.base import TransformerMixin
from upath import UPath as Path

from drevalpy.components.core.tuning.config import HPOConfig, validate_hpo_metric
from drevalpy.components.core.tuning.drp_hyperparameters import (
    default_hyperparameters_for_drp_model,
    has_tunable_hyperparameters,
    public_hyperparameters_from_config,
    structured_space_for_drp_model,
    tuned_config_for_drp_model,
)
from drevalpy.components.core.tuning.hpo_runtime import build_optuna_objective, run_optuna_study
from drevalpy.data.structures import EntityScope
from drevalpy.data.structures.mudataset import MuDataset
from drevalpy.log import get_logger
from drevalpy.models.drp_model import DRPModel

logger = get_logger(__name__)


def _is_valid_score(value: float) -> bool:
    return bool(np.isfinite(value))


def mu_hpam_tune(
    *,
    model_class: type[DRPModel],
    mudataset: MuDataset,
    train_scope: EntityScope,
    val_scope: EntityScope,
    early_stopping_scope: EntityScope | None,
    response_transformation: TransformerMixin | None = None,
    metric: str = "RMSE",
    model_checkpoint_dir: str | Path | None = None,
    hpo_config: HPOConfig | None = None,
    split_index: int | None = None,
    wandb_project: str | None = None,
    wandb_base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tune hyperparameters using MuDataset + EntityScope with Optuna.

    :param model_class: Model class to tune.
    :param mudataset: Full dataset with all features.
    :param train_scope: Training EntityScope.
    :param val_scope: Validation EntityScope for scoring.
    :param early_stopping_scope: Optional early-stopping scope.
    :param response_transformation: Optional response transformer.
    :param metric: Metric to optimize.
    :param model_checkpoint_dir: Directory for model checkpoints.
    :param hpo_config: HPO configuration.
    :param split_index: CV fold index for W&B logging.
    :param wandb_project: W&B project name.
    :param wandb_base_config: Base W&B config merged per trial.
    :returns: Best flat hyperparameter mapping.
    """
    validate_hpo_metric(metric)
    cfg = hpo_config or HPOConfig.from_metric(metric)
    if cfg.metric != metric:
        msg = f"HPOConfig.metric ({cfg.metric!r}) must match metric argument ({metric!r})"
        raise ValueError(msg)

    structured_space = structured_space_for_drp_model(model_class)
    if not structured_space or not has_tunable_hyperparameters(model_class):
        return model_class.get_default_hyperparameters()
    if cfg.n_trials == 0:
        return model_class.get_default_hyperparameters()

    model_name = model_class.get_model_name()
    objective = build_optuna_objective(
        model_class=model_class,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=early_stopping_scope,
        response_transformation=response_transformation,
        metric=metric,
        structured_space=structured_space,
        model_checkpoint_dir=model_checkpoint_dir,
        cfg=cfg,
        wandb_project=wandb_project,
        wandb_base_config=wandb_base_config,
        split_index=split_index,
        model_name=model_name,
    )

    study = run_optuna_study(objective=objective, cfg=cfg)

    best_trial = study.best_trial if study.best_trial is not None else None
    if best_trial is None or not _is_valid_score(best_trial.value):
        warnings.warn(
            "Optuna tuning did not find a valid configuration; using defaults.",
            stacklevel=2,
        )
        return default_hyperparameters_for_drp_model(model_class)

    best_config = best_trial.params
    best_model_config = tuned_config_for_drp_model(model_class, best_config)
    if best_model_config is None:
        return dict(best_config)
    return public_hyperparameters_from_config(best_model_config)
