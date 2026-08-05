"""Ray Tune execution helpers for component HPO."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from sklearn.base import TransformerMixin

from drevalpy.components.tuning.config import HPOConfig
from drevalpy.components.tuning.drp_hyperparameters import (
    construct_drp_model_from_config,
    tuned_config_for_drp_model,
)
from drevalpy.components.tuning.search_space import dict_to_ray_space
from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models.drp_model import DRPModel

logger = logging.getLogger(__name__)


def current_trial_id() -> str:
    """Current trial id.

    :returns: Result.
    """
    try:
        from ray import tune

        trial_id = tune.get_context().get_trial_id()
        if trial_id:
            return str(trial_id)
    except Exception as exc:
        logger.debug("Ray Tune trial context unavailable: %s", exc)
    return "unknown"


def trial_checkpoint_dir(base_dir: str) -> str:
    """Trial checkpoint dir.

    :param base_dir: base dir.
    :returns: Result.
    """
    path = os.path.join(base_dir, f"trial_{current_trial_id()}")
    os.makedirs(path, exist_ok=True)
    return path


def build_ray_trainable(
    *,
    model_class: type[DRPModel],
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    response_transformation: TransformerMixin | None,
    metric: str,
    path_data: str,
    model_checkpoint_dir: str,
    cfg: HPOConfig,
    wandb_project: str | None,
    wandb_base_config: dict[str, Any] | None,
    split_index: int | None,
    model_name: str,
) -> Callable[[dict[str, Any]], None]:
    """Build ray trainable.

    :param model_class: model class.
    :param train_dataset: train dataset.
    :param validation_dataset: validation dataset.
    :param early_stopping_dataset: early stopping dataset.
    :param response_transformation: response transformation.
    :param metric: metric.
    :param path_data: path data.
    :param model_checkpoint_dir: model checkpoint dir.
    :param cfg: cfg.
    :param wandb_project: wandb project.
    :param wandb_base_config: wandb base config.
    :param split_index: split index.
    :param model_name: model name.
    :returns: Result.
    """
    from ray import tune

    from drevalpy import experiment

    def _construct_trial_model(sampled: dict[str, Any]) -> DRPModel:
        trial_config = tuned_config_for_drp_model(model_class, sampled)
        if trial_config is None:
            return model_class(sampled)
        return construct_drp_model_from_config(model_class, trial_config)

    def _evaluate_sample(trial_model: DRPModel) -> float:
        trial_dir = trial_checkpoint_dir(model_checkpoint_dir)
        result = experiment.train_and_evaluate(
            model=trial_model,
            path_data=path_data,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            early_stopping_dataset=early_stopping_dataset,
            metric=metric,
            response_transformation=response_transformation,
            model_checkpoint_dir=trial_dir,
        )
        return float(result[metric])

    def trainable(sampled: dict[str, Any]) -> None:
        """Trainable.

        :param sampled: sampled.
        """
        try:
            score = _evaluate_sample(_construct_trial_model(sampled))
            tune.report({metric: score})
        except Exception:
            logger.exception("Ray/Optuna trial failed")
            tune.report({metric: float("nan")})

    def trainable_with_wandb(sampled: dict[str, Any]) -> None:
        """Trainable with wandb.

        :param sampled: sampled.
        """
        if wandb_project is None:
            trainable(sampled)
            return
        trial_model = _construct_trial_model(sampled)
        trial_model._in_hyperparameter_tuning = True
        trial_id = current_trial_id()
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
        trial_run_name = model_name
        if split_index is not None:
            trial_run_name += f"_split_{split_index}"
        trial_run_name += f"_trial_{trial_id}"
        trial_model.init_wandb(
            project=wandb_project,
            config=trial_run_config,
            name=trial_run_name,
            tags=[model_name, "hpam_tuning", "ray", "optuna"],
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

    return trainable_with_wandb if wandb_project is not None else trainable


def run_ray_tuner(
    *,
    trainable_fn: Callable[[dict[str, Any]], None],
    structured_space: dict[str, Any],
    cfg: HPOConfig,
) -> Any:
    """Run ray tuner.

    :param trainable_fn: trainable fn.
    :param structured_space: structured space.
    :param cfg: cfg.
    :returns: Result.
    """
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch

    param_space = dict_to_ray_space(structured_space)
    search_alg = OptunaSearch(metric=cfg.metric, mode=cfg.mode, seed=cfg.random_state)
    wrapped = tune.with_resources(trainable_fn, resources=cfg.resources_per_trial)
    tuner = tune.Tuner(
        wrapped,
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
    return tuner.fit()
