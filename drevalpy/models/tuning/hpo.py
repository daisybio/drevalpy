"""Optuna hyperparameter optimization for DRPModel experiments."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from upath import UPath as Path

from drevalpy.log import get_logger
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.tuning.config import HPOConfig, validate_hpo_metric
from drevalpy.models.tuning.config_resolution import (
    has_tunable_hyperparameters,
    tuned_config_for_drp_model,
)
from drevalpy.models.tuning.hpo_runtime import (
    _construct_trial_model,
    _mu_evaluate_trial_all_metrics,
    run_optuna_study,
)
from drevalpy.models.tuning.public_flat import public_hyperparameters_from_config
from drevalpy.models.tuning.search_space import sample_from_optuna_trial
from drevalpy.types import SplitMask
from drevalpy.types.data.dataset import Dataset

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin

logger = get_logger(__name__)


class HPOTrialsFailedError(RuntimeError):
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


def _is_valid_score(value: float) -> bool:
    return bool(np.isfinite(value))


def hpam_tune(
    *,
    model_class: type[DRPModel],
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
    precomputed_only: bool = False,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, float], np.ndarray]]]:
    """Tune hyperparameters using Dataset + SplitMask with Optuna.

    Returns the best hyperparameter mapping and per-trial results with all metrics.

    :param model_class: Model class to tune.
    :param mudataset: Full dataset with all features.
    :param train_scope: Training SplitMask.
    :param val_scope: Validation SplitMask for scoring.
    :param early_stopping_scope: Optional early-stopping scope.
    :param response_transformation: Optional response transformer.
    :param metric: Metric to optimize.
    :param model_checkpoint_dir: Directory for model checkpoints.
    :param hpo_config: HPO configuration.
    :param split_index: CV fold index for W&B logging.
    :param wandb_project: W&B project name.
    :param wandb_base_config: Base W&B config merged per trial.
    :param precomputed_only: When True, restrict fixed featurizer HP params to
        stored variants. (Search space restriction is a TODO; plumbing is in place.)
    :returns: Tuple of (best_params, trial_results) where trial_results is a
        list of (hyperparameters, metrics_dict, predictions) tuples for each completed trial.
    :raises ValueError: Raised when ``hpo_config.metric`` disagrees with ``metric``.
    :raises HPOTrialsFailedError: Raised when every trial raised an exception, so no
        tuning information was produced.
    """
    validate_hpo_metric(metric)
    cfg = hpo_config or HPOConfig.from_metric(metric)
    if cfg.metric != metric:
        msg = f"HPOConfig.metric ({cfg.metric!r}) must match metric argument ({metric!r})"
        raise ValueError(msg)

    structured_space = model_class.get_structured_hyperparameter_space()
    if not structured_space or not has_tunable_hyperparameters(model_class):
        return model_class.get_default_hyperparameters(), []
    if cfg.n_trials == 0:
        return model_class.get_default_hyperparameters(), []

    # TODO: When precomputed_only is True, query list_stored_variants(mdata) for each
    # precomputable featurizer in the model config and restrict its HP params in
    # structured_space to categorical choices over stored variant values.
    _ = precomputed_only

    model_name = model_class.get_model_name()
    all_trial_data: list[tuple[dict[str, Any], dict[str, float], np.ndarray]] = []
    failures = _TrialFailures()

    def objective_with_metrics(trial: Any) -> float:
        sampled = sample_from_optuna_trial(trial, structured_space)
        trial_model = _construct_trial_model(model_class, sampled)

        try:
            metrics, predictions = _mu_evaluate_trial_all_metrics(
                trial_model,
                mudataset=mudataset,
                train_scope=train_scope,
                val_scope=val_scope,
                early_stopping_scope=early_stopping_scope,
                response_transformation=response_transformation,
                model_checkpoint_dir=model_checkpoint_dir,
                trial_number=trial.number,
            )
        except Exception as exc:
            # Returning NaN rather than re-raising keeps the study running, so a
            # search space with a few invalid corners still tunes. Whether the
            # whole run was a fault is decided once, in _report_trial_failures.
            failures.record(exc)
            logger.exception("Optuna trial %d failed", trial.number)
            return float("nan")

        if metrics:
            all_trial_data.append((sampled, metrics, predictions))

        if wandb_project:
            _log_trial_to_wandb(
                wandb_project=wandb_project,
                wandb_base_config=wandb_base_config,
                model_name=model_name,
                split_index=split_index,
                trial_number=trial.number,
                sampled=sampled,
                metrics=metrics,
                metric=metric,
            )

        target = metrics.get(metric, float("nan"))
        return target if _is_valid_score(target) else float("nan")

    study = run_optuna_study(objective=objective_with_metrics, cfg=cfg)
    _report_trial_failures(study, failures)
    best_params = _resolve_best_params(study, model_class)
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


def _resolve_best_params(study, model_class: type[DRPModel]) -> dict[str, Any]:
    """Extract best hyperparameters from a completed Optuna study."""
    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None

    if best_trial is None or not _is_valid_score(best_trial.value):
        warnings.warn(
            "Optuna tuning did not find a valid configuration; using defaults.",
            stacklevel=2,
        )
        return model_class.get_default_hyperparameters()

    best_config = best_trial.params
    best_model_config = tuned_config_for_drp_model(model_class, best_config)
    if best_model_config is None:
        return dict(best_config)
    return public_hyperparameters_from_config(best_model_config)


def _log_trial_to_wandb(
    *,
    wandb_project: str,
    wandb_base_config: dict[str, Any] | None,
    model_name: str,
    split_index: int | None,
    trial_number: int,
    sampled: dict[str, Any],
    metrics: dict[str, float],
    metric: str,
) -> None:
    """Log a single HPO trial to W&B."""
    try:
        import wandb
    except ImportError:
        return

    run_name = f"{model_name}_split{split_index}_trial{trial_number}"
    run_config = dict(wandb_base_config or {})
    run_config.update(sampled)
    run = wandb.init(
        project=wandb_project,
        name=run_name,
        config=run_config,
        reinit=True,
    )
    if run is not None:
        target_value = metrics.get(metric, float("nan"))
        wandb.log({"hpo_metric": target_value, **metrics})
        run.finish()
