"""Ray Tune execution helpers for component HPO."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.base import TransformerMixin
from upath import UPath as Path

from drevalpy.components.tuning.config import HPOConfig
from drevalpy.components.tuning.drp_hyperparameters import (
    construct_drp_model_from_config,
    tuned_config_for_drp_model,
)
from drevalpy.components.tuning.search_space import dict_to_ray_space
from drevalpy.data.structures.mudataset import MuDataset
from drevalpy.data.structures import EntityScope
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


def _extract_ground_truth(mudataset: MuDataset, scope: EntityScope) -> np.ndarray:
    """Extract ground truth response values from MuDataset for the given scope.

    :param mudataset: Source of response values.
    :param scope: EntityScope with cell-line/drug indices.
    :returns: 1-D array of non-NaN ground-truth response values.
    """
    response_matrix = mudataset.response_matrix
    cl_idx = scope.cell_lines
    dr_idx = scope.drugs

    if dr_idx is None:
        sub_matrix = response_matrix[np.ix_(cl_idx, np.arange(response_matrix.shape[1]))]
        values = sub_matrix[~np.isnan(sub_matrix)]
    elif len(cl_idx) == response_matrix.shape[0] or (
        len(cl_idx) > 0 and np.array_equal(cl_idx, np.arange(response_matrix.shape[0]))
    ):
        sub_matrix = response_matrix[np.ix_(cl_idx, dr_idx)]
        values = sub_matrix[~np.isnan(sub_matrix)]
    else:
        responses = response_matrix[cl_idx, dr_idx]
        values = responses[~np.isnan(responses)]

    return values.astype(np.float64)


def _mu_evaluate_trial_model(
    trial_model: DRPModel,
    *,
    metric: str,
    mudataset: MuDataset,
    train_scope: EntityScope,
    val_scope: EntityScope,
    early_stopping_scope: EntityScope | None,
    response_transformation: TransformerMixin | None,
    model_checkpoint_dir: str | Path | None,
) -> float:
    """Train a trial model and compute a validation metric using MuDataset + EntityScope."""
    from drevalpy.evaluation import AVAILABLE_METRICS

    trial_dir = trial_checkpoint_dir(model_checkpoint_dir)
    from drevalpy.utils.checkpoints import checkpoint_dir_or_temporary

    with checkpoint_dir_or_temporary(trial_dir) as checkpoint_dir:
        trial_model.train(
            mudataset=mudataset,
            scope=train_scope,
            early_stopping_scope=early_stopping_scope,
            model_checkpoint_dir=checkpoint_dir,
        )

    predictions = trial_model.predict(mudataset=mudataset, scope=val_scope)

    if response_transformation is not None:
        predictions = response_transformation.inverse_transform(predictions.reshape(-1, 1)).ravel()

    ground_truth = _extract_ground_truth(mudataset, val_scope)

    if len(predictions) != len(ground_truth):
        min_len = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_len]
        ground_truth = ground_truth[:min_len]

    if len(predictions) == 0:
        return float("nan")

    metric_fn = AVAILABLE_METRICS.get(metric)
    if metric_fn is None:
        return float("nan")
    return float(metric_fn(y_pred=predictions, y_true=ground_truth))


def mu_build_ray_trainable(
    *,
    model_class: type[DRPModel],
    mudataset: MuDataset,
    train_scope: EntityScope,
    val_scope: EntityScope,
    early_stopping_scope: EntityScope | None,
    response_transformation: TransformerMixin | None,
    metric: str,
    model_checkpoint_dir: str | Path | None,
    cfg: HPOConfig,
    wandb_project: str | None,
    wandb_base_config: dict[str, Any] | None,
    split_index: int | None,
    model_name: str,
) -> Callable[[dict[str, Any]], None]:
    """Build a Ray Tune trainable using MuDataset + EntityScope.

    :param model_class: Model class to tune.
    :param mudataset: Full dataset with all features.
    :param train_scope: Training EntityScope.
    :param val_scope: Validation EntityScope for scoring.
    :param early_stopping_scope: Optional early-stopping scope.
    :param response_transformation: Optional response transformer.
    :param metric: Metric to optimize.
    :param model_checkpoint_dir: Directory for model checkpoints.
    :param cfg: HPO configuration.
    :param wandb_project: W&B project name.
    :param wandb_base_config: Base W&B config merged per trial.
    :param split_index: CV fold index for W&B logging.
    :param model_name: Model name for logging.
    :returns: Callable trainable for Ray Tune.
    """

    def trainable(sampled: dict[str, Any]) -> None:
        try:
            trial_model = _construct_trial_model(model_class, sampled)
            score = _mu_evaluate_trial_model(
                trial_model,
                metric=metric,
                mudataset=mudataset,
                train_scope=train_scope,
                val_scope=val_scope,
                early_stopping_scope=early_stopping_scope,
                response_transformation=response_transformation,
                model_checkpoint_dir=model_checkpoint_dir,
            )
            _report_trial_score(metric, score)
        except Exception:
            _report_trial_failure(metric)

    def trainable_with_wandb(sampled: dict[str, Any]) -> None:
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
            score = _mu_evaluate_trial_model(
                trial_model,
                metric=metric,
                mudataset=mudataset,
                train_scope=train_scope,
                val_scope=val_scope,
                early_stopping_scope=early_stopping_scope,
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
