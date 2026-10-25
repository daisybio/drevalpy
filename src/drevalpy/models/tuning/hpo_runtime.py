"""Optuna-based hyperparameter optimization runtime for component HPO."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from upath import UPath as Path

from drevalpy.evaluation._scoring import run_metrics
from drevalpy.models.spec import ModelSpec
from drevalpy.models.tuning.config import HPOConfig
from drevalpy.models.tuning.search_space import resolve_model_config
from drevalpy.types import SplitMask
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.dataset import Dataset

if TYPE_CHECKING:
    import optuna

    from drevalpy.models.drp_model import DRPModel
    from drevalpy.models.response_transformation import ResponseTransformer


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
    drawn: dict[str, Any],
    response_transformation: ResponseTransformer | None = None,
) -> DRPModel:
    from drevalpy.models.drp_model import DRPModel
    from drevalpy.models.tuning.public_flat import public_hyperparameters_from_config

    config = resolve_model_config(model_spec.config, drawn)
    return DRPModel(
        model_spec, public_hyperparameters_from_config(config), response_transformation=response_transformation
    )


def _extract_ground_truth(dataset: Dataset, scope: SplitMask) -> np.ndarray:
    """Extract ground truth response values from Dataset for the given scope.

    :param dataset: Source of response values.
    :param scope: SplitMask with 2D pair array.
    :returns: 1-D scope-aligned array, including missing response values.
    """
    return ResponseBatch.from_dataset(dataset, scope).response


def _mu_evaluate_trial_all_metrics(
    trial_model: DRPModel,
    *,
    dataset: Dataset,
    train_scope: SplitMask,
    val_scope: SplitMask,
    early_stopping_scope: SplitMask | None,
    model_checkpoint_dir: str | Path | None,
    trial_number: int = 0,
    n_jobs: int = 1,
) -> tuple[dict[str, float], np.ndarray]:
    """Train a trial model, then score its validation predictions on every metric.

    :param trial_model: Unfitted model built from the hyperparameters of this trial.
    :param dataset: Full dataset with all features.
    :param train_scope: Training SplitMask.
    :param val_scope: Validation SplitMask for scoring.
    :param early_stopping_scope: Optional early-stopping scope.
    :param model_checkpoint_dir: Directory for per-trial checkpoints; ``None`` uses a temporary directory.
    :param trial_number: Optuna trial number, used to name the trial's checkpoint directory.
    :param n_jobs: Worker processes a single-drug predictor fits its per-drug models in.
    :returns: Tuple of (metrics_dict, predictions_array).
    """
    from contextlib import nullcontext

    trial_dir = _trial_checkpoint_dir(model_checkpoint_dir, trial_number)
    context = nullcontext(trial_dir) if trial_dir is not None else tempfile.TemporaryDirectory()
    with context as checkpoint_dir:
        trial_model.train(
            dataset,
            train_scope,
            early_stopping_scope=early_stopping_scope,
            model_checkpoint_dir=checkpoint_dir,
            n_jobs=n_jobs,
        )
    predictions = trial_model.predict(dataset, val_scope)
    return run_metrics(predictions, _extract_ground_truth(dataset, val_scope)), predictions


def run_optuna_study(
    *,
    objective: Callable[[optuna.Trial], float],
    cfg: HPOConfig,
) -> optuna.Study:
    """Create and run an Optuna study.

    :param objective: Function that Optuna calls once per trial; it returns the score to optimize.
    :param cfg: HPO configuration giving the number of trials, the optimization direction and the sampler seed.
    :returns: Completed Optuna study.
    """
    import optuna

    sampler = optuna.samplers.TPESampler(seed=cfg.random_state)
    study = optuna.create_study(direction=cfg.direction, sampler=sampler)
    study.optimize(objective, n_trials=cfg.n_trials)
    return study
