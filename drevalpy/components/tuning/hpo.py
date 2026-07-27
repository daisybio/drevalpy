"""Ray Tune / Optuna hyperparameter optimization for DRPModel experiments."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import numpy as np
from sklearn.base import TransformerMixin

from drevalpy.components.tuning.config import HPOConfig, validate_hpo_metric
from drevalpy.components.tuning.drp_hyperparameters import (
    construct_drp_model_from_config,
    default_hyperparameters_for_drp_model,
    has_tunable_hyperparameters,
    public_hyperparameters_from_config,
    structured_space_for_drp_model,
    tuned_config_for_drp_model,
)
from drevalpy.components.tuning.search_space import dict_to_ray_space
from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models.drp_model import DRPModel

logger = logging.getLogger(__name__)


def _current_trial_id() -> str:
    try:
        from ray import tune

        trial_id = tune.get_context().get_trial_id()
        if trial_id:
            return str(trial_id)
    except Exception as exc:
        logger.debug("Ray Tune trial context unavailable: %s", exc)
    return "unknown"


def _trial_checkpoint_dir(base_dir: str) -> str:
    path = os.path.join(base_dir, f"trial_{_current_trial_id()}")
    os.makedirs(path, exist_ok=True)
    return path


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


def _select_best_result(results: Any, cfg: HPOConfig) -> Any | None:
    try:
        best_result = results.get_best_result(metric=cfg.metric, mode=cfg.mode)
    except Exception as exc:
        logger.warning("Ray Tune get_best_result failed: %s", exc)
        best_result = None

    if best_result is not None:
        score = _metric_value(best_result.metrics, cfg.metric)
        if _is_valid_score(score) and best_result.config:
            return best_result

    best_candidate = None
    best_score = float("inf") if cfg.mode == "min" else float("-inf")
    try:
        trial_results = list(results)
    except TypeError:
        trial_results = []

    for result in trial_results:
        score = _metric_value(result.metrics, cfg.metric)
        if not _is_valid_score(score) or not result.config:
            continue
        if cfg.mode == "min" and score < best_score:
            best_score = score
            best_candidate = result
        elif cfg.mode == "max" and score > best_score:
            best_score = score
            best_candidate = result

    return best_candidate


def hpam_tune(
    model: DRPModel,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    *,
    model_class: type[DRPModel],
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
    from drevalpy import experiment

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
        from ray import tune
        from ray.tune.search.optuna import OptunaSearch
    except ImportError as exc:
        msg = "Ray Tune with Optuna requires ray[tune] and optuna to be installed"
        raise ImportError(msg) from exc

    ray_initialized_here = not ray.is_initialized()
    model._in_hyperparameter_tuning = True
    try:
        if ray_initialized_here:
            ray.init(ignore_reinit_error=True)

        param_space = dict_to_ray_space(structured_space)

        def _construct_trial_model(sampled: dict[str, Any]) -> DRPModel:
            trial_config = tuned_config_for_drp_model(model_class, sampled)
            if trial_config is None:
                return model_class(sampled)
            return construct_drp_model_from_config(model_class, trial_config)

        def _evaluate_sample(trial_model: DRPModel) -> float:
            trial_checkpoint_dir = _trial_checkpoint_dir(model_checkpoint_dir)
            result = experiment.train_and_evaluate(
                model=trial_model,
                path_data=path_data,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                early_stopping_dataset=early_stopping_dataset,
                metric=metric,
                response_transformation=response_transformation,
                model_checkpoint_dir=trial_checkpoint_dir,
            )
            return float(result[metric])

        def trainable(sampled: dict[str, Any]) -> None:
            try:
                score = _evaluate_sample(_construct_trial_model(sampled))
                tune.report({metric: score})
            except Exception:
                logger.exception("Ray/Optuna trial failed")
                tune.report({metric: float("nan")})

        def trainable_with_wandb(sampled: dict[str, Any]) -> None:
            if wandb_project is None:
                trainable(sampled)
                return
            trial_model = _construct_trial_model(sampled)
            trial_id = _current_trial_id()
            trial_run_config: dict[str, Any] = {
                "phase": "hyperparameter_tuning",
                "hpo_backend": "ray",
                "search_alg": cfg.search_alg,
                "hpo_num_samples": cfg.n_trials,
                "hyperparameters": trial_model.hyperparameters,
                "trial_id": trial_id,
            }
            if wandb_base_config is not None:
                trial_run_config = {**wandb_base_config, **trial_run_config}
            trial_run_name = model.get_model_name()
            if split_index is not None:
                trial_run_name += f"_split_{split_index}"
            trial_run_name += f"_trial_{trial_id}"
            trial_model.init_wandb(
                project=wandb_project,
                config=trial_run_config,
                name=trial_run_name,
                tags=[model.get_model_name(), "hpam_tuning", "ray", "optuna"],
                finish_previous=True,
            )
            try:
                score = _evaluate_sample(trial_model)
                tune.report({metric: score})
            except Exception:
                logger.exception("Ray/Optuna trial failed")
                tune.report({metric: float("nan")})
            finally:
                if trial_model.is_wandb_enabled():
                    trial_model.finish_wandb()

        search_alg = OptunaSearch(metric=cfg.metric, mode=cfg.mode, seed=cfg.random_state)
        trainable_fn = tune.with_resources(
            trainable_with_wandb if wandb_project is not None else trainable,
            resources=cfg.resources_per_trial,
        )
        tuner = tune.Tuner(
            trainable_fn,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                metric=cfg.metric,
                mode=cfg.mode,
                num_samples=cfg.n_trials,
                search_alg=search_alg,
            ),
            run_config=(
                tune.RunConfig(
                    storage_path=cfg.storage_path,
                    name="hpam_tuning",
                )
                if cfg.storage_path is not None
                else None
            ),
        )
        results = tuner.fit()
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
        model._in_hyperparameter_tuning = False
        if ray_initialized_here and ray.is_initialized():
            ray.shutdown()
