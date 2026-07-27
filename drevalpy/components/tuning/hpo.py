"""Ray Tune / Optuna hyperparameter optimization for DRPModel experiments."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
from sklearn.base import TransformerMixin

from drevalpy.components.tuning.config import HPOConfig, validate_hpo_metric
from drevalpy.components.tuning.drp_hyperparameters import (
    default_hyperparameters_for_drp_model,
    has_tunable_hyperparameters,
    public_hyperparameters_from_config,
    structured_space_for_drp_model,
    tuned_config_for_drp_model,
)
from drevalpy.components.tuning.hpo_runtime import build_ray_trainable, run_ray_tuner
from drevalpy.datasets.dataset import DrugResponseDataset
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


def tune_fold(
    model_class: type[DRPModel],
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    *,
    response_transformation: TransformerMixin | None = None,
    metric: str = "RMSE",
    path_data: str = "data",
    model_checkpoint_dir: str = "TEMPORARY",
    hpo_config: HPOConfig | None = None,
    split_index: int | None = None,
    wandb_project: str | None = None,
    wandb_base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tune hyperparameters for one fold and return the best flat public dict."""
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
        split_index=split_index,
        wandb_project=wandb_project,
        wandb_base_config=wandb_base_config,
    )


def hpam_tune(
    *,
    model_class: type[DRPModel],
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    response_transformation: TransformerMixin | None = None,
    metric: str = "RMSE",
    path_data: str = "data",
    model_checkpoint_dir: str = "TEMPORARY",
    hpo_config: HPOConfig | None = None,
    split_index: int | None = None,
    wandb_project: str | None = None,
    wandb_base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tune hyperparameters with Ray Tune and OptunaSearch over structured spaces."""
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
        trainable = build_ray_trainable(
            model_class=model_class,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            early_stopping_dataset=early_stopping_dataset,
            response_transformation=response_transformation,
            metric=metric,
            path_data=path_data,
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
