"""Optuna hyperparameter tuning of one model on one fold.

An untuned run, zero trials or an empty search space returns the model recipe's values
without Optuna. Each trial trains a fresh ``DRPModel`` on the training scope and is scored
on the validation scope with every metric. The study optimizes the chosen metric.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy.errors import DrEvalError
from drevalpy.evaluation import Metric
from drevalpy.evaluation._scoring import evaluate
from drevalpy.execution._tracking import log_summary, require_tracking, tracking_run
from drevalpy.log import get_logger
from drevalpy.models.config.immutable import thaw_value
from drevalpy.models.spec import ModelSpec
from drevalpy.models.tuning.search_space import draw
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.data.response_table import ResponseTable
from drevalpy.types.results.run import TrialResult

if TYPE_CHECKING:
    import optuna

    from drevalpy.models.response_transformation import ResponseTransformer
    from drevalpy.types.data.fold import Fold

logger = get_logger(__name__)


class HPOTrialsFailedError(DrEvalError, RuntimeError):
    """Every hyperparameter trial raised, so tuning produced no information.

    Falling back to defaults here would report a tuning result for a run where
    nothing was tuned, and bury the real cause - a missing native library, a bad
    search space, a bug in the model - behind a later, unrelated traceback.
    """


def hpam_tune(
    spec: ModelSpec,
    dataset: Dataset,
    fold: Fold,
    *,
    tuning: bool,
    metric: Metric,
    n_trials: int,
    sampler_seed: int,
    response_transformation: ResponseTransformer | None = None,
    wandb_project: str | None = None,
    wandb_base_config: Mapping[str, Any] | None = None,
    n_jobs: int = 1,
) -> tuple[dict[str, Any], list[TrialResult]]:
    """Tune the model recipe with Optuna using the prepared fold.

    Raises ``ImportError`` if *wandb_project* is set and ``wandb`` is not installed, and
    ``HPOTrialsFailedError`` if every trial raised, so no tuning information was produced.

    :param spec: Named, unfitted model recipe.
    :param dataset: Full dataset with all features.
    :param fold: Prepared fold with training, validation, and optional stopping masks.
    :param tuning: Whether to search the hyperparameter space.
    :param metric: Validation metric to optimize.
    :param n_trials: Number of Optuna trials.
    :param sampler_seed: Seed for the Optuna sampler.
    :param response_transformation: Optional unfitted response transformation, cloned for every trial.
    :param wandb_project: W&B project to log one run per trial to; ``None`` disables tracking.
    :param wandb_base_config: Base W&B config; each trial's drawn hyperparameters are added to it.
    :param n_jobs: Worker processes a single-drug predictor fits its per-drug models in, for every trial.
    :returns: The complete best values under qualified keys (the spec's defaults when nothing is tuned),
        and one ``TrialResult`` per completed trial.
    """
    structured_space = spec.hyperparameter_space
    if not (tuning and n_trials and structured_space):
        return _defaults(spec), []

    require_tracking(wandb_project)
    fold_index = fold.metadata.get("fold_index", 0)
    ground_truth = ResponseTable.from_dataset(dataset, fold.val).response
    trials: list[TrialResult] = []
    failures: list[Exception] = []

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
                model.train(dataset, fold.train, early_stopping_scope=fold.stopping, n_jobs=n_jobs)
                predictions = model.predict(dataset, fold.val)
                metrics = evaluate(predictions, ground_truth)
                log_summary(
                    {"hpo_metric": metrics.get(metric, float("nan")), **{f"val_{k}": v for k, v in metrics.items()}}
                )
        except Exception as exc:
            # Returning NaN rather than re-raising keeps the study running, so a
            # search space with a few invalid corners still tunes. Whether the
            # whole run was a fault is decided once, in _report_trial_failures.
            failures.append(exc)
            logger.exception("Optuna trial %d failed", trial.number)
            return float("nan")

        if metrics:
            trials.append(TrialResult(drawn, metrics, metric.value, predictions))
        target = metrics.get(metric, float("nan"))
        return target if np.isfinite(target) else float("nan")

    study = _run_study(objective, metric=metric, n_trials=n_trials, sampler_seed=sampler_seed)
    _report_trial_failures(study, failures)
    return _best_params(study, spec), trials


def _run_study(
    objective: Callable[[optuna.Trial], float], *, metric: Metric, n_trials: int, sampler_seed: int
) -> optuna.Study:
    """Run a seeded TPE study in the direction of the metric.

    :param objective: Function Optuna calls once per trial; it returns the score.
    :param metric: Validation metric to optimize.
    :param n_trials: Number of Optuna trials.
    :param sampler_seed: Seed for the Optuna sampler.
    :returns: Completed Optuna study.
    """
    import optuna

    direction = "minimize" if metric.direction == "min" else "maximize"
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study


def _report_trial_failures(study: optuna.Study, failures: list[Exception]) -> None:
    """Raise when no trial survived; warn when only some did.

    :param study: Completed Optuna study, used for the number of trials actually run.
    :param failures: Exceptions raised by the failed trials, in order.
    :raises HPOTrialsFailedError: Raised when every trial raised, chaining the first
        exception so the original cause heads the traceback.
    """
    if not failures:
        return

    total = len(study.trials)
    if len(failures) < total:
        logger.warning(
            "%d of %d hyperparameter trials failed; tuning used the %d that survived",
            len(failures),
            total,
            total - len(failures),
        )
        return

    msg = f"All {total} hyperparameter trials failed with {type(failures[0]).__name__}: {failures[0]}"
    raise HPOTrialsFailedError(msg) from failures[0]


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
