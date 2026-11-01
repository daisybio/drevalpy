"""Optuna hyperparameter tuning of one model on one fold.

:func:`hpam_tune` reads every tuning decision (trial count, metric, sampler seed and the scopes)
from the fold's ``ExecutionSpec``, so it is called the same way whether or not the run tunes: an
untuned run, zero trials or an empty search space return the spec's defaults without Optuna.
Each trial trains a fresh ``DRPModel`` on the training scope and is scored on the validation
scope with every metric; the study minimizes or maximizes the execution's metric.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy.errors import DrEvalError
from drevalpy.evaluation import Metric
from drevalpy.evaluation._scoring import evaluate
from drevalpy.experiment._tracking import log_summary, require_tracking, tracking_run
from drevalpy.log import get_logger
from drevalpy.models.config.immutable import thaw_value
from drevalpy.models.spec import ModelSpec
from drevalpy.models.tuning.search_space import draw
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.results.trial import TrialResult

if TYPE_CHECKING:
    import optuna

    from drevalpy.experiment._execution import ExecutionSpec
    from drevalpy.models.response_transformation import ResponseTransformer

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


def hpam_tune(
    execution: ExecutionSpec,
    dataset: Dataset,
    *,
    response_transformation: ResponseTransformer | None = None,
    wandb_project: str | None = None,
    wandb_base_config: Mapping[str, Any] | None = None,
    n_jobs: int = 1,
) -> tuple[dict[str, Any], list[TrialResult]]:
    """Tune the hyperparameters of the execution's model with Optuna.

    Raises ``ImportError`` if *wandb_project* is set and ``wandb`` is not installed, and
    ``HPOTrialsFailedError`` if every trial raised, so no tuning information was produced.

    :param execution: Fold decisions: the model, its scopes, and the tuning metric, trial count and sampler seed.
    :param dataset: Full dataset with all features.
    :param response_transformation: Optional unfitted response transformation, cloned for every trial.
    :param wandb_project: W&B project to log one run per trial to; ``None`` disables tracking.
    :param wandb_base_config: Base W&B config; each trial's drawn hyperparameters are added to it.
    :param n_jobs: Worker processes a single-drug predictor fits its per-drug models in, for every trial.
    :returns: The complete best values under qualified keys (the spec's defaults when nothing is tuned),
        and one ``TrialResult`` per completed trial.
    """
    spec = execution.model
    structured_space = spec.hyperparameter_space
    if not (execution.hyperparameter_tuning and execution.hpo_num_trials and structured_space):
        return _defaults(spec), []

    require_tracking(wandb_project)
    metric, scopes, fold_index = execution.hpo_metric, execution.scopes, execution.identity.fold_index
    ground_truth = ResponseBatch.from_dataset(dataset, scopes.validation).response
    trials: list[TrialResult] = []
    failures = _TrialFailures()

    def objective(trial: optuna.Trial) -> float:
        from drevalpy.models.drp_model import DRPModel

        drawn = draw(trial, structured_space)
        model = DRPModel(spec.with_hyperparameters(drawn), response_transformation=response_transformation)
        try:
            with tracking_run(
                wandb_project,
                name=f"{spec.name}_split{fold_index}_trial{trial.number}",
                config={**(wandb_base_config or {}), **drawn},
                tags=[spec.name, "trial"],
            ):
                model.train(dataset, scopes.training, early_stopping_scope=scopes.stopping, n_jobs=n_jobs)
                predictions = model.predict(dataset, scopes.validation)
                metrics = evaluate(predictions, ground_truth)
                log_summary(
                    {"hpo_metric": metrics.get(metric, float("nan")), **{f"val_{k}": v for k, v in metrics.items()}}
                )
        except Exception as exc:
            # Returning NaN rather than re-raising keeps the study running, so a
            # search space with a few invalid corners still tunes. Whether the
            # whole run was a fault is decided once, in _report_trial_failures.
            failures.record(exc)
            logger.exception("Optuna trial %d failed", trial.number)
            return float("nan")

        if metrics:
            trials.append(TrialResult(drawn, metrics, metric, predictions))
        target = metrics.get(metric, float("nan"))
        return target if np.isfinite(target) else float("nan")

    study = _run_study(objective, execution)
    _report_trial_failures(study, failures)
    return _best_params(study, spec), trials


def _run_study(objective: Callable[[optuna.Trial], float], execution: ExecutionSpec) -> optuna.Study:
    """Run a seeded TPE study of ``execution.hpo_num_trials`` trials in the direction of the metric.

    :param objective: Function Optuna calls once per trial; it returns the score.
    :param execution: Fold decisions giving the metric, trial count and sampler seed.
    :returns: Completed Optuna study.
    """
    import optuna

    direction = "minimize" if Metric(execution.hpo_metric).direction == "min" else "maximize"
    sampler = optuna.samplers.TPESampler(seed=execution.hpo_random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=execution.hpo_num_trials)
    return study


def _report_trial_failures(study: optuna.Study, failures: _TrialFailures) -> None:
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


def _best_params(study: optuna.Study, spec: ModelSpec) -> dict[str, Any]:
    """Return the complete values of the best trial, or the defaults when no trial scored."""
    from optuna.trial import FixedTrial

    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None

    if best_trial is None or best_trial.value is None or not np.isfinite(best_trial.value):
        warnings.warn("Optuna tuning did not find a valid configuration; using defaults.", stacklevel=2)
        return _defaults(spec)

    # ``best_trial.params`` holds what Optuna sampled (a ``pow2`` exponent), not what
    # the trials trained with; replaying it through the draw code decodes it identically.
    best = draw(FixedTrial(best_trial.params), spec.hyperparameter_space)
    return _defaults(spec.with_hyperparameters(best))


def _defaults(spec: ModelSpec) -> dict[str, Any]:
    """Return an independent copy of the spec's values under qualified keys."""
    return thaw_value(dict(spec.hyperparameters))
