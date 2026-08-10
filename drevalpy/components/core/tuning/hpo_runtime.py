"""Optuna-based hyperparameter optimization runtime for component HPO."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from typing import Any

import numpy as np
import optuna
from sklearn.base import TransformerMixin
from upath import UPath as Path

from drevalpy.components.core.tuning.config import HPOConfig
from drevalpy.components.core.tuning.drp_hyperparameters import (
    construct_drp_model_from_config,
    tuned_config_for_drp_model,
)
from drevalpy.components.core.tuning.search_space import sample_from_optuna_trial
from drevalpy.log import get_logger
from drevalpy.models.drp_model import DRPModel
from drevalpy.types import SplitMask
from drevalpy.types.data.dataset import Dataset

logger = get_logger(__name__)


def _trial_checkpoint_dir(base_dir: str | Path | None, trial_number: int) -> Path | None:
    """Return a per-trial subdirectory of *base_dir*.

    :param base_dir: Root checkpoint directory, or ``None`` for a temporary one.
    :param trial_number: Optuna trial number.
    :returns: The trial subdirectory, or ``None`` when a temporary one should be used.
    """
    if base_dir is None:
        return None
    path = Path(base_dir) / f"trial_{trial_number}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _construct_trial_model(model_class: type[DRPModel], sampled: dict[str, Any]) -> DRPModel:
    trial_config = tuned_config_for_drp_model(model_class, sampled)
    if trial_config is None:
        return model_class(sampled)
    return construct_drp_model_from_config(model_class, trial_config)


def _extract_ground_truth(mudataset: Dataset, scope: SplitMask) -> np.ndarray:
    """Extract ground truth response values from Dataset for the given scope.

    :param mudataset: Source of response values.
    :param scope: SplitMask with 2D pair array.
    :returns: 1-D array of non-NaN ground-truth response values.
    """
    response_matrix = mudataset.response_matrix
    pairs = scope.pairs
    cl_idx = pairs[:, 0]
    dr_idx = pairs[:, 1]

    responses = response_matrix[cl_idx, dr_idx]
    values = responses[~np.isnan(responses)]

    return values.astype(np.float64)


def _mu_evaluate_trial_model(
    trial_model: DRPModel,
    *,
    metric: str,
    mudataset: Dataset,
    train_scope: SplitMask,
    val_scope: SplitMask,
    early_stopping_scope: SplitMask | None,
    response_transformation: TransformerMixin | None,
    model_checkpoint_dir: str | Path | None,
    trial_number: int = 0,
) -> float:
    """Train a trial model and compute a validation metric using Dataset + SplitMask."""
    from drevalpy.evaluation import AVAILABLE_METRICS

    trial_dir = _trial_checkpoint_dir(model_checkpoint_dir, trial_number)
    if trial_dir is not None:
        trial_model.train(
            mudataset=mudataset,
            scope=train_scope,
            early_stopping_scope=early_stopping_scope,
            model_checkpoint_dir=str(trial_dir),
        )
    else:
        with tempfile.TemporaryDirectory() as checkpoint_dir:
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

    # Filter out NaN predictions (from pairs with missing features)
    valid = ~np.isnan(predictions) & ~np.isnan(ground_truth)
    if not valid.any():
        return float("nan")
    predictions = predictions[valid]
    ground_truth = ground_truth[valid]

    metric_fn = AVAILABLE_METRICS.get(metric)
    if metric_fn is None:
        return float("nan")
    return float(metric_fn(y_pred=predictions, y_true=ground_truth))


def _mu_evaluate_trial_all_metrics(
    trial_model: DRPModel,
    *,
    mudataset: Dataset,
    train_scope: SplitMask,
    val_scope: SplitMask,
    early_stopping_scope: SplitMask | None,
    response_transformation: TransformerMixin | None,
    model_checkpoint_dir: str | Path | None,
    trial_number: int = 0,
) -> tuple[dict[str, float], np.ndarray]:
    """Train a trial model and compute all validation metrics.

    :returns: Tuple of (metrics_dict, predictions_array).
    """
    from drevalpy.evaluation import AVAILABLE_METRICS

    trial_dir = _trial_checkpoint_dir(model_checkpoint_dir, trial_number)
    if trial_dir is not None:
        trial_model.train(
            mudataset=mudataset,
            scope=train_scope,
            early_stopping_scope=early_stopping_scope,
            model_checkpoint_dir=str(trial_dir),
        )
    else:
        with tempfile.TemporaryDirectory() as checkpoint_dir:
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
        return {}, predictions

    valid = ~np.isnan(predictions) & ~np.isnan(ground_truth)
    if not valid.any():
        return {}, predictions

    metrics: dict[str, float] = {}
    for name, fn in AVAILABLE_METRICS.items():
        metrics[name] = float(fn(y_pred=predictions[valid], y_true=ground_truth[valid]))
    return metrics, predictions


def _wandb_trial_run_config(
    *,
    trial_model: DRPModel,
    cfg: HPOConfig,
    wandb_base_config: dict[str, Any] | None,
    trial_number: int,
) -> dict[str, Any]:
    trial_run_config: dict[str, Any] = {
        "phase": "hyperparameter_tuning",
        "hpo_backend": "optuna",
        "hpo_num_samples": cfg.n_trials,
        "hyperparameters": trial_model.hyperparameters,
        "trial_number": trial_number,
    }
    if wandb_base_config is not None:
        trial_run_config = {**wandb_base_config, **trial_run_config}
    return trial_run_config


def _wandb_trial_run_name(*, model_name: str, split_index: int | None, trial_number: int) -> str:
    trial_run_name = model_name
    if split_index is not None:
        trial_run_name += f"_split_{split_index}"
    return f"{trial_run_name}_trial_{trial_number}"


def _init_trial_wandb(
    trial_model: DRPModel,
    *,
    wandb_project: str,
    wandb_base_config: dict[str, Any] | None,
    cfg: HPOConfig,
    model_name: str,
    split_index: int | None,
    trial_number: int,
) -> None:
    trial_model.init_wandb(
        project=wandb_project,
        config=_wandb_trial_run_config(
            trial_model=trial_model,
            cfg=cfg,
            wandb_base_config=wandb_base_config,
            trial_number=trial_number,
        ),
        name=_wandb_trial_run_name(model_name=model_name, split_index=split_index, trial_number=trial_number),
        tags=[model_name, "hpam_tuning", "optuna"],
        finish_previous=True,
    )


def _optuna_objective(
    trial: optuna.Trial,
    *,
    model_class: type[DRPModel],
    mudataset: Dataset,
    train_scope: SplitMask,
    val_scope: SplitMask,
    early_stopping_scope: SplitMask | None,
    response_transformation: TransformerMixin | None,
    metric: str,
    structured_space: dict[str, Any],
    model_checkpoint_dir: str | Path | None,
    cfg: HPOConfig,
    wandb_project: str | None,
    wandb_base_config: dict[str, Any] | None,
    split_index: int | None,
    model_name: str,
) -> float:
    """Optuna objective function: sample params, train, evaluate, return score."""
    sampled = sample_from_optuna_trial(trial, structured_space)
    trial_model = _construct_trial_model(model_class, sampled)

    if wandb_project is not None:
        trial_model._in_hyperparameter_tuning = True
        _init_trial_wandb(
            trial_model,
            wandb_project=wandb_project,
            wandb_base_config=wandb_base_config,
            cfg=cfg,
            model_name=model_name,
            split_index=split_index,
            trial_number=trial.number,
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
            trial_number=trial.number,
        )
    except Exception:
        logger.exception("Optuna trial %d failed", trial.number)
        score = float("nan")
    finally:
        if wandb_project is not None and trial_model.is_wandb_enabled():
            trial_model.finish_wandb()

    return score


def build_optuna_objective(
    *,
    model_class: type[DRPModel],
    mudataset: Dataset,
    train_scope: SplitMask,
    val_scope: SplitMask,
    early_stopping_scope: SplitMask | None,
    response_transformation: TransformerMixin | None,
    metric: str,
    structured_space: dict[str, Any],
    model_checkpoint_dir: str | Path | None,
    cfg: HPOConfig,
    wandb_project: str | None,
    wandb_base_config: dict[str, Any] | None,
    split_index: int | None,
    model_name: str,
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective function closure.

    :param model_class: Model class to tune.
    :param mudataset: Full dataset with all features.
    :param train_scope: Training SplitMask.
    :param val_scope: Validation SplitMask for scoring.
    :param early_stopping_scope: Optional early-stopping scope.
    :param response_transformation: Optional response transformer.
    :param metric: Metric to optimize.
    :param structured_space: Structured hyperparameter search space.
    :param model_checkpoint_dir: Directory for model checkpoints.
    :param cfg: HPO configuration.
    :param wandb_project: W&B project name.
    :param wandb_base_config: Base W&B config merged per trial.
    :param split_index: CV fold index for W&B logging.
    :param model_name: Model name for logging.
    :returns: Callable objective for ``study.optimize()``.
    """

    def objective(trial: optuna.Trial) -> float:
        return _optuna_objective(
            trial,
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

    return objective


def run_optuna_study(
    *,
    objective: Callable[[optuna.Trial], float],
    cfg: HPOConfig,
) -> optuna.Study:
    """Create and run an Optuna study.

    :param objective: The objective function.
    :param cfg: HPO configuration.
    :returns: Completed Optuna study.
    """
    direction = "minimize" if cfg.mode == "min" else "maximize"
    sampler = optuna.samplers.TPESampler(seed=cfg.random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=cfg.n_trials)
    return study
