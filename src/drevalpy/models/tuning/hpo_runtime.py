"""Optuna-based hyperparameter optimization runtime for component HPO."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from upath import UPath as Path

from drevalpy.evaluation import score_predictions
from drevalpy.models.spec import ModelSpec
from drevalpy.models.tuning.config import HPOConfig
from drevalpy.models.tuning.search_space import resolve_model_config
from drevalpy.types import SplitMask
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.dataset import Dataset

if TYPE_CHECKING:
    import optuna
    from sklearn.base import TransformerMixin

    from drevalpy.models.drp_model import DRPModel


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
