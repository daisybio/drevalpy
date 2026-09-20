"""Optuna-based hyperparameter optimization runtime for component HPO."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from upath import UPath as Path

from drevalpy.evaluation import score_predictions
from drevalpy.log import get_logger
from drevalpy.models.spec import ModelSpec
from drevalpy.models.tuning.config import HPOConfig
from drevalpy.models.tuning.search_space import resolve_model_config, sample_from_optuna_trial
from drevalpy.types import SplitMask
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.dataset import Dataset

if TYPE_CHECKING:
    import optuna
    from sklearn.base import TransformerMixin

    from drevalpy.models.drp_model import DRPModel

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


def _construct_trial_model(
    model_spec: ModelSpec,
    sampled: dict[str, Any],
    response_transformation: TransformerMixin | None = None,
) -> DRPModel:
    from drevalpy.models.drp_model import DRPModel
    from drevalpy.models.tuning.public_flat import public_hyperparameters_from_config

    config = resolve_model_config(model_spec.config, sampled)
    return DRPModel(
        model_spec, public_hyperparameters_from_config(config), response_transformation=response_transformation
    )


def _extract_ground_truth(mudataset: Dataset, scope: SplitMask) -> np.ndarray:
    """Extract ground truth response values from Dataset for the given scope.

    :param mudataset: Source of response values.
    :param scope: SplitMask with 2D pair array.
    :returns: 1-D scope-aligned array, including missing response values.
    """
    return ResponseBatch.from_dataset(mudataset, scope).response


def _mu_evaluate_trial_model(
    trial_model: DRPModel,
    *,
    metric: str,
    mudataset: Dataset,
    train_scope: SplitMask,
    val_scope: SplitMask,
    early_stopping_scope: SplitMask | None,
    model_checkpoint_dir: str | Path | None,
    trial_number: int = 0,
) -> float:
    """Train a trial model and compute a validation metric using Dataset + SplitMask."""
    metrics, _ = _mu_evaluate_trial_all_metrics(
        trial_model,
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        early_stopping_scope=early_stopping_scope,
        model_checkpoint_dir=model_checkpoint_dir,
        trial_number=trial_number,
    )
    return metrics.get(metric, float("nan"))


def _mu_evaluate_trial_all_metrics(
    trial_model: DRPModel,
    *,
    mudataset: Dataset,
    train_scope: SplitMask,
    val_scope: SplitMask,
    early_stopping_scope: SplitMask | None,
    model_checkpoint_dir: str | Path | None,
    trial_number: int = 0,
) -> tuple[dict[str, float], np.ndarray]:
    """Train a trial model and compute all validation metrics.

    :returns: Tuple of (metrics_dict, predictions_array).
    """
    from contextlib import nullcontext

    trial_dir = _trial_checkpoint_dir(model_checkpoint_dir, trial_number)
    context = nullcontext(trial_dir) if trial_dir is not None else tempfile.TemporaryDirectory()
    with context as checkpoint_dir:
        trial_model.train(
            mudataset,
            train_scope,
            early_stopping_scope=early_stopping_scope,
            model_checkpoint_dir=checkpoint_dir,
        )
    predictions = trial_model.predict(mudataset, val_scope)
    return score_predictions(predictions, _extract_ground_truth(mudataset, val_scope), empty_as_dict=True), predictions


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
    model_spec: ModelSpec,
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
    trial_model = _construct_trial_model(model_spec, sampled, response_transformation)

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
    import optuna

    direction = "minimize" if cfg.mode == "min" else "maximize"
    sampler = optuna.samplers.TPESampler(seed=cfg.random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=cfg.n_trials)
    return study
