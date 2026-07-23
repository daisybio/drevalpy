"""Ray Tune / Optuna hyperparameter optimization for DRPModel experiments."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.base import TransformerMixin

from drevalpy.components.tuning.config import HPOConfig
from drevalpy.components.tuning.drp_hyperparameters import (
    build_drp_model_from_config,
    default_hyperparameters_for_drp_model,
    has_tunable_hyperparameters,
    public_hyperparameters_from_config,
    structured_space_for_drp_model,
    tuned_config_for_drp_model,
)
from drevalpy.components.tuning.search_space import dict_to_ray_space
from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models.drp_model import DRPModel


def hpam_tune_ray_optuna(
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

    cfg = hpo_config or HPOConfig.from_metric(metric)
    structured_space = structured_space_for_drp_model(model_class)
    if not structured_space or not has_tunable_hyperparameters(model_class):
        return model_class.get_default_hyperparameters()

    if cfg.n_trials <= 1:
        return model_class.get_default_hyperparameters()

    try:
        import ray
        from ray import tune
        from ray.tune.search.optuna import OptunaSearch
    except ImportError as exc:
        msg = "Ray Tune with Optuna requires ray[tune] and optuna to be installed"
        raise ImportError(msg) from exc

    model._in_hyperparameter_tuning = True
    param_space = dict_to_ray_space(structured_space)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    def _evaluate_sample(sampled: dict[str, Any], trial_model: DRPModel) -> float:
        trial_config = tuned_config_for_drp_model(model_class, sampled)
        if trial_config is None:
            trial_model.build_model(sampled)
        else:
            build_drp_model_from_config(trial_model, trial_config)
        result = experiment.train_and_evaluate(
            model=trial_model,
            hpams=trial_model.hyperparameters,
            path_data=path_data,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            early_stopping_dataset=early_stopping_dataset,
            metric=metric,
            response_transformation=response_transformation,
            model_checkpoint_dir=model_checkpoint_dir,
        )
        return float(result[metric])

    def trainable(sampled: dict[str, Any]) -> None:
        trial_model = model_class()
        try:
            score = _evaluate_sample(sampled, trial_model)
            tune.report({metric: score})
        except Exception as exc:
            print("Trial failed:", exc)
            tune.report({metric: float("nan")})

    def trainable_with_wandb(sampled: dict[str, Any]) -> None:
        if wandb_project is None:
            trainable(sampled)
            return
        trial_model = model_class()
        trial_config = tuned_config_for_drp_model(model_class, sampled)
        hpams = public_hyperparameters_from_config(trial_config) if trial_config is not None else dict(sampled)
        trial_run_config: dict[str, Any] = {
            "phase": "hyperparameter_tuning",
            "hpo_backend": "ray",
            "search_alg": cfg.search_alg,
            "hpo_num_samples": cfg.n_trials,
            "hyperparameters": hpams,
        }
        if wandb_base_config is not None:
            trial_run_config = {**wandb_base_config, **trial_run_config}
        trial_run_name = model.get_model_name()
        if split_index is not None:
            trial_run_name += f"_split_{split_index}"
        trial_run_name += "_trial"
        trial_model.init_wandb(
            project=wandb_project,
            config=trial_run_config,
            name=trial_run_name,
            tags=[model.get_model_name(), "hpam_tuning", "ray", "optuna"],
            finish_previous=True,
        )
        try:
            score = _evaluate_sample(sampled, trial_model)
            tune.report({metric: score})
        except Exception as exc:
            print("Trial failed:", exc)
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
    best_result = results.get_best_result(metric=cfg.metric, mode=cfg.mode)
    best_config = best_result.config or {}
    best_metrics = best_result.metrics or {}
    model._in_hyperparameter_tuning = False

    if not best_config or np.isnan(best_metrics.get(cfg.metric, float("nan"))):
        warnings.warn(
            "Ray/Optuna tuning did not find a valid configuration; using defaults.",
            stacklevel=2,
        )
        return default_hyperparameters_for_drp_model(model_class)
    best_model_config = tuned_config_for_drp_model(model_class, best_config)
    if best_model_config is None:
        return dict(best_config)
    return public_hyperparameters_from_config(best_model_config)
