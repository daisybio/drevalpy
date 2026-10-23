"""Optuna hyperparameter optimization for DRPModel experiments."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from upath import UPath as Path

from drevalpy.errors import DrEvalError
from drevalpy.experiment._tracking import log_summary, require_tracking, tracking_run
from drevalpy.log import get_logger
from drevalpy.models.spec import ModelSpec
from drevalpy.models.tuning.config import HPOConfig, validate_hpo_metric
from drevalpy.models.tuning.hpo_runtime import (
    _construct_trial_model,
    _mu_evaluate_trial_all_metrics,
    run_optuna_study,
)
from drevalpy.models.tuning.public_flat import public_hyperparameters_from_config
from drevalpy.models.tuning.search_space import draw_from_optuna_trial, resolve_model_config
from drevalpy.types import SplitMask
from drevalpy.types.data.dataset import Dataset

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin

logger = get_logger(__name__)


class HPOTrialsFailedError(DrEvalError, RuntimeError):
    """Every hyperparameter trial raised, so tuning produced no information.

    Falling back to defaults here would report a tuning result for a run where
    nothing was tuned, and bury the real cause - a missing native library, a bad
    search space, a bug in the model - behind a later, unrelated traceback.
    """


@dataclass
class _TrialFailures:
    """Count of trials that raised, plus the first exception, for reporting."""

    count: int = 0
    first: BaseException | None = None

    def record(self, exc: BaseException) -> None:
        """Record a raising trial, keeping the first exception as the cause.

        :param exc: Exception raised by the trial.
        """
        self.count += 1
        if self.first is None:
            self.first = exc


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _is_valid_score(value: float) -> bool:
    return bool(np.isfinite(value))


def hpam_tune(
    *,
    model_spec: ModelSpec,
    mudataset: Dataset,
    train_scope: SplitMask,
    val_scope: SplitMask,
    early_stopping_scope: SplitMask | None,
    response_transformation: TransformerMixin | None = None,
    metric: str = "RMSE",
    model_checkpoint_dir: str | Path | None = None,
    hpo_config: HPOConfig | None = None,
    split_index: int | None = None,
    wandb_project: str | None = None,
    wandb_base_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, float], np.ndarray]]]:
    """Tune hyperparameters using Dataset + SplitMask with Optuna.

    Returns the best hyperparameter mapping and per-trial results with all metrics.

    :param model_spec: Named model recipe to tune.
    :param mudataset: Full dataset with all features.
    :param train_scope: Training SplitMask.
    :param val_scope: Validation SplitMask for scoring.
    :param early_stopping_scope: Optional early-stopping scope.
    :param response_transformation: Optional response transformer.
    :param metric: Metric to optimize.
    :param model_checkpoint_dir: Directory for model checkpoints.
    :param hpo_config: HPO configuration.
    :param split_index: Fold index, used in the W&B run names.
    :param wandb_project: W&B project to log one run per trial to; ``None`` disables tracking.
    :param wandb_base_config: Base W&B config; each trial's drawn hyperparameters are added to it.
    :returns: Tuple of (best_params, trial_results) where trial_results is a
        list of (hyperparameters, metrics_dict, predictions) tuples for each completed trial.
    :raises ValueError: Raised when ``hpo_config.metric`` disagrees with ``metric``.
    :raises ImportError: Raised when ``wandb_project`` is set and ``wandb`` is not installed.
    :raises HPOTrialsFailedError: Raised when every trial raised an exception, so no
        tuning information was produced.
    """
    validate_hpo_metric(metric)
    cfg = hpo_config or HPOConfig.from_metric(metric)
    if cfg.metric != metric:
        msg = f"HPOConfig.metric ({cfg.metric!r}) must match metric argument ({metric!r})"
        raise ValueError(msg)

    structured_space = model_spec.hyperparameter_space
    if not structured_space:
        return model_spec.default_hyperparameters, []
    if cfg.n_trials == 0:
        return model_spec.default_hyperparameters, []

    require_tracking(wandb_project)
    model_name = model_spec.name
    all_trial_data: list[tuple[dict[str, Any], dict[str, float], np.ndarray]] = []
    failures = _TrialFailures()

    def objective_with_metrics(trial: Any) -> float:
        drawn = draw_from_optuna_trial(trial, structured_space)
        trial_model = _construct_trial_model(model_spec, drawn, response_transformation)

        try:
            with tracking_run(
                wandb_project,
                name=f"{model_name}_split{split_index}_trial{trial.number}",
                config={**(wandb_base_config or {}), **drawn},
                tags=[model_name, "trial"],
            ):
                metrics, predictions = _mu_evaluate_trial_all_metrics(
                    trial_model,
                    mudataset=mudataset,
                    train_scope=train_scope,
                    val_scope=val_scope,
                    early_stopping_scope=early_stopping_scope,
                    model_checkpoint_dir=model_checkpoint_dir,
                    trial_number=trial.number,
                )
                log_summary({"hpo_metric": metrics.get(metric, float("nan")), **_prefixed("val_", metrics)})
        except Exception as exc:
            # Returning NaN rather than re-raising keeps the study running, so a
            # search space with a few invalid corners still tunes. Whether the
            # whole run was a fault is decided once, in _report_trial_failures.
            failures.record(exc)
            logger.exception("Optuna trial %d failed", trial.number)
            return float("nan")

        if metrics:
            all_trial_data.append((drawn, metrics, predictions))

        target = metrics.get(metric, float("nan"))
        return target if _is_valid_score(target) else float("nan")

    study = run_optuna_study(objective=objective_with_metrics, cfg=cfg)
    _report_trial_failures(study, failures)
    best_params = _resolve_best_params(study, model_spec)
    return best_params, all_trial_data


def _report_trial_failures(study, failures: _TrialFailures) -> None:
    """Raise when no trial survived; warn when only some did.

    :param study: Completed Optuna study, used for the number of trials actually run.
    :param failures: Recorded trial failures.
    :raises HPOTrialsFailedError: Raised when every trial raised, chaining the first
        exception so the original cause heads the traceback.
    """
    if failures.count == 0:
        return

    total = len(study.trials)
    if failures.count < total:
        logger.warning(
            "%d of %d hyperparameter trials failed; tuning used the %d that survived",
            failures.count,
            total,
            total - failures.count,
        )
        return

    msg = f"All {total} hyperparameter trials failed with {type(failures.first).__name__}: {failures.first}"
    raise HPOTrialsFailedError(msg) from failures.first


def _resolve_best_params(study, model_spec: ModelSpec) -> dict[str, Any]:
    """Extract best hyperparameters from a completed Optuna study."""
    from optuna.trial import FixedTrial

    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None

    if best_trial is None or not _is_valid_score(best_trial.value):
        warnings.warn(
            "Optuna tuning did not find a valid configuration; using defaults.",
            stacklevel=2,
        )
        return model_spec.default_hyperparameters

    # ``best_trial.params`` holds what Optuna sampled (a ``pow2`` exponent), not what
    # the trials trained with; replaying it through the draw code decodes it identically.
    best_config = draw_from_optuna_trial(FixedTrial(best_trial.params), model_spec.hyperparameter_space)
    best_model_config = resolve_model_config(model_spec.config, best_config)
    return public_hyperparameters_from_config(best_model_config)
