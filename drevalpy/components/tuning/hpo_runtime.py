"""Ray Tune execution helpers for component HPO."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sklearn.base import TransformerMixin

from drevalpy.components.tuning.config import HPOConfig
from drevalpy.components.tuning.drp_hyperparameters import (
    construct_drp_model_from_config,
    tuned_config_for_drp_model,
)
from drevalpy.components.tuning.search_space import dict_to_ray_space
from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.models.drp_model import DRPModel
from drevalpy.utils.checkpoints import resolve_checkpoint_dir

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


def trial_checkpoint_dir(base_dir: str | Path | None) -> Path | None:
    """Return a per-trial subdirectory of *base_dir*.

    :param base_dir: Root checkpoint directory, or ``None`` for a temporary one.

    :returns: The trial subdirectory, or ``None`` when a temporary one should be used.
    """
    resolved = resolve_checkpoint_dir(base_dir)
    if resolved is None:
        return None
    path = resolved / f"trial_{current_trial_id()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _construct_trial_model(model_class: type[DRPModel], sampled: dict[str, Any]) -> DRPModel:
    trial_config = tuned_config_for_drp_model(model_class, sampled)
    if trial_config is None:
        return model_class(sampled)
    return construct_drp_model_from_config(model_class, trial_config)


def _evaluate_trial_model(
    trial_model: DRPModel,
    *,
    metric: str,
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    response_transformation: TransformerMixin | None,
    model_checkpoint_dir: str | Path | None,
) -> float:
    from drevalpy import experiment

    trial_dir = trial_checkpoint_dir(model_checkpoint_dir)
    result = experiment.train_and_evaluate(
        model=trial_model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        early_stopping_dataset=early_stopping_dataset,
        metric=metric,
        response_transformation=response_transformation,
        model_checkpoint_dir=trial_dir,
    )
    return float(result[metric])


def _report_trial_score(metric: str, score: float) -> None:
    from ray import tune

    tune.report({metric: score})


def _report_trial_failure(metric: str) -> None:
    logger.exception("Ray/Optuna trial failed")
    _report_trial_score(metric, float("nan"))


def _wandb_trial_run_config(
    *,
    trial_model: DRPModel,
    cfg: HPOConfig,
    wandb_base_config: dict[str, Any] | None,
    trial_id: str,
) -> dict[str, Any]:
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
    return trial_run_config


def _wandb_trial_run_name(*, model_name: str, split_index: int | None, trial_id: str) -> str:
    trial_run_name = model_name
    if split_index is not None:
        trial_run_name += f"_split_{split_index}"
    return f"{trial_run_name}_trial_{trial_id}"


def _init_trial_wandb(
    trial_model: DRPModel,
    *,
    wandb_project: str,
    wandb_base_config: dict[str, Any] | None,
    cfg: HPOConfig,
    model_name: str,
    split_index: int | None,
) -> None:
    trial_id = current_trial_id()
    trial_model.init_wandb(
        project=wandb_project,
        config=_wandb_trial_run_config(
            trial_model=trial_model,
            cfg=cfg,
            wandb_base_config=wandb_base_config,
            trial_id=trial_id,
        ),
        name=_wandb_trial_run_name(model_name=model_name, split_index=split_index, trial_id=trial_id),
        tags=[model_name, "hpam_tuning", "ray", "optuna"],
        finish_previous=True,
    )


def build_ray_trainable(
    *,
    model_class: type[DRPModel],
    train_dataset: DrugResponseDataset,
    validation_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
    response_transformation: TransformerMixin | None,
    metric: str,
    model_checkpoint_dir: str | Path | None,
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
    :param model_checkpoint_dir: Directory for model checkpoints, or ``None`` for a temporary one.
    :param cfg: cfg.
    :param wandb_project: wandb project.
    :param wandb_base_config: wandb base config.
    :param split_index: split index.
    :param model_name: model name.
    :returns: Result.
    """

    def trainable(sampled: dict[str, Any]) -> None:
        """Trainable.

        :param sampled: sampled.
        """
        try:
            trial_model = _construct_trial_model(model_class, sampled)
            score = _evaluate_trial_model(
                trial_model,
                metric=metric,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                early_stopping_dataset=early_stopping_dataset,
                response_transformation=response_transformation,
                model_checkpoint_dir=model_checkpoint_dir,
            )
            _report_trial_score(metric, score)
        except Exception:
            _report_trial_failure(metric)

    def trainable_with_wandb(sampled: dict[str, Any]) -> None:
        """Trainable with wandb.

        :param sampled: sampled.
        """
        if wandb_project is None:
            trainable(sampled)
            return
        trial_model = _construct_trial_model(model_class, sampled)
        trial_model._in_hyperparameter_tuning = True
        _init_trial_wandb(
            trial_model,
            wandb_project=wandb_project,
            wandb_base_config=wandb_base_config,
            cfg=cfg,
            model_name=model_name,
            split_index=split_index,
        )
        try:
            score = _evaluate_trial_model(
                trial_model,
                metric=metric,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                early_stopping_dataset=early_stopping_dataset,
                response_transformation=response_transformation,
                model_checkpoint_dir=model_checkpoint_dir,
            )
            _report_trial_score(metric, score)
        except Exception:
            _report_trial_failure(metric)
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
