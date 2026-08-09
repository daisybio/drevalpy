"""Ray Tune / Optuna hyperparameter optimization for DRPModel experiments."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
from sklearn.base import TransformerMixin
from upath import UPath as Path

from drevalpy.components.tuning.config import HPOConfig, validate_hpo_metric
from drevalpy.components.tuning.drp_hyperparameters import (
    default_hyperparameters_for_drp_model,
    has_tunable_hyperparameters,
    public_hyperparameters_from_config,
    structured_space_for_drp_model,
    tuned_config_for_drp_model,
)
from drevalpy.components.tuning.hpo_runtime import mu_build_ray_trainable, run_ray_tuner
from drevalpy.data.structures.mudataset import MuDataset
from drevalpy.data.structures.splitting import EntityScope
from drevalpy.models.drp_model import DRPModel

logger = logging.getLogger(__name__)


def _metric_value(metrics: dict[str, Any] | None, metric: str) -> float:
    if not metrics:
        return float("nan")
    value = metrics.get(metric, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_valid_score(value: float) -> bool:
    return bool(np.isfinite(value))


def _trial_is_usable(result: Any, cfg: HPOConfig) -> bool:
    score = _metric_value(result.metrics, cfg.metric)
    return _is_valid_score(score) and bool(result.config)


def _better_trial_score(score: float, best_score: float, mode: str) -> bool:
    if mode == "min":
        return score < best_score
    return score > best_score


def _best_result_from_scan(results: Any, cfg: HPOConfig) -> Any | None:
    best_candidate = None
    best_score = float("inf") if cfg.mode == "min" else float("-inf")
    try:
        trial_results = list(results)
    except TypeError:
        return None

    for result in trial_results:
        if not _trial_is_usable(result, cfg):
            continue
        score = _metric_value(result.metrics, cfg.metric)
        if _better_trial_score(score, best_score, cfg.mode):
            best_score = score
            best_candidate = result
    return best_candidate


def _select_best_result(results: Any, cfg: HPOConfig) -> Any | None:
    try:
        best_result = results.get_best_result(metric=cfg.metric, mode=cfg.mode)
    except Exception as exc:
        logger.warning("Ray Tune get_best_result failed: %s", exc)
        best_result = None

    if best_result is not None and _trial_is_usable(best_result, cfg):
        return best_result

    return _best_result_from_scan(results, cfg)


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
    """Tune hyperparameters using MuDataset + EntityScope.

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

    try:
        import ray
    except ImportError as exc:
        msg = "Ray Tune with Optuna requires ray[tune] and optuna to be installed"
        raise ImportError(msg) from exc

    ray_initialized_here = not ray.is_initialized()
    try:
        if ray_initialized_here:
            ray.init(ignore_reinit_error=True)

        model_name = model_class.get_model_name()
        trainable = mu_build_ray_trainable(
            model_class=model_class,
            mudataset=mudataset,
            train_scope=train_scope,
            val_scope=val_scope,
            early_stopping_scope=early_stopping_scope,
            response_transformation=response_transformation,
            metric=metric,
            model_checkpoint_dir=model_checkpoint_dir,
            cfg=cfg,
            wandb_project=wandb_project,
            wandb_base_config=wandb_base_config,
            split_index=split_index,
            model_name=model_name,
        )
        results = run_ray_tuner(trainable_fn=trainable, structured_space=structured_space, cfg=cfg)
        best_result = _select_best_result(results, cfg)
        if best_result is None:
            warnings.warn(
                "Ray/Optuna tuning did not find a valid configuration; using defaults.",
                stacklevel=2,
            )
            return default_hyperparameters_for_drp_model(model_class)

        best_config = best_result.config or {}
        best_model_config = tuned_config_for_drp_model(model_class, best_config)
        if best_model_config is None:
            return dict(best_config)
        return public_hyperparameters_from_config(best_model_config)
    finally:
        if ray_initialized_here and ray.is_initialized():
            ray.shutdown()
