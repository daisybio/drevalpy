"""Single model + single fold execution unit.

:func:`_fit_fold` is the one path from a model, a dataset and a fold to a ``RunResult``:
``drevalpy.experiment``, ``drevalpy.run`` and ``drevalpy run`` all go through it. It fits and
predicts, and trains the normalization reference ``NaiveMeanEffectsPredictor``
without the response transformation, so a fold and the same settings give the same ``RunResult``
whichever entry point ran them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from drevalpy._utils.n_jobs import check_n_jobs
from drevalpy.execution._fit import fit
from drevalpy.log import get_logger
from drevalpy.models.spec import ModelSpec
from drevalpy.types import Fold
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.results._normalize import NORMALIZATION_REFERENCE
from drevalpy.types.results.run import RunResult

if TYPE_CHECKING:
    from drevalpy.models.response_transformation import ResponseTransformer

logger = get_logger(__name__)


def run(
    model_spec: ModelSpec,
    dataset: Dataset,
    fold: Fold,
    *,
    hyperparameter_tuning: bool = True,
    response_transformation: ResponseTransformer | None = None,
    hpo_metric: str = "RMSE",
    hpo_num_trials: int = 16,
    hpo_random_state: int | None = None,
    seed: int = 42,
    measure: str | None = None,
    wandb_project: str | None = None,
    n_jobs: int = 1,
) -> RunResult:
    """Train a single model on a single fold and predict on the test set.

    :param model_spec: Named, unfitted model recipe.
    :param dataset: Full dataset with all features.
    :param fold: The fold to train, tune and test on.
    :param hyperparameter_tuning: Whether to run HPO.
    :param response_transformation: Optional unfitted response transformation prototype (any sklearn-compatible
        transformer); a clone is fitted per scope and the caller's instance is left untouched. The normalization
        reference ``NaiveMeanEffectsPredictor`` is always fitted on the untransformed response, as in
        :func:`drevalpy.experiment`.
    :param hpo_metric: Metric to optimize during HPO.
    :param hpo_num_trials: Number of HPO trials.
    :param hpo_random_state: Random seed for the HPO sampler; defaults to ``seed``.
    :param seed: Root seed; every HPO run and final fit is seeded from it and the execution identity.
    :param measure: Response measure to train and score on, as for :func:`drevalpy.experiment`. ``None`` takes the
        measure the fold was split on (its ``measure`` metadata), else the dataset's. A fold split on another
        measure is rejected.
    :param wandb_project: Weights & Biases project to log the trials and the final fit to; ``None`` disables
        tracking.
    :param n_jobs: Number of worker processes a single-drug model fits its per-drug models in, during tuning and in the
        final fit; ``-1`` uses all CPUs. The result does not depend on it, and other models ignore it.
    :returns: RunResult with predictions, ground truth, and metrics.
    """
    check_n_jobs(n_jobs)
    measure = measure or fold.metadata.get("measure")
    return _fit_fold(
        model_spec,
        dataset if measure is None else dataset.with_measure(measure),
        fold,
        hyperparameter_tuning=hyperparameter_tuning,
        response_transformation=response_transformation,
        hpo_metric=hpo_metric,
        hpo_num_trials=hpo_num_trials,
        hpo_random_state=hpo_random_state,
        seed=seed,
        wandb_project=wandb_project,
        n_jobs=n_jobs,
    )


def _fit_fold(
    model_spec: ModelSpec,
    dataset: Dataset,
    fold: Fold,
    *,
    hyperparameter_tuning: bool,
    response_transformation: ResponseTransformer | None,
    hpo_metric: str,
    hpo_num_trials: int,
    hpo_random_state: int | None,
    seed: int,
    wandb_project: str | None,
    n_jobs: int,
    hyperparameters: Mapping[str, Any] | None = None,
) -> RunResult:
    """Fit and predict one fold; the callers have already checked *n_jobs*.

    :param model_spec: Named, unfitted model recipe.
    :param dataset: Dataset on the measure to train and score on.
    :param fold: The fold to train, tune and test on.
    :param hyperparameter_tuning: Whether to run HPO; ignored when *hyperparameters* is given.
    :param response_transformation: Optional unfitted response transformation prototype; the normalization
        reference trains without it.
    :param hpo_metric: Metric to optimize during HPO.
    :param hpo_num_trials: Number of HPO trials.
    :param hpo_random_state: Random seed for the HPO sampler; defaults to ``seed``.
    :param seed: Root seed of the HPO runs and the final fit.
    :param wandb_project: Weights & Biases project; ``None`` disables tracking.
    :param n_jobs: Worker processes a single-drug model fits its per-drug models in.
    :param hyperparameters: Values to fit with instead of tuning, such as the ``best_hyperparameters`` of an
        earlier run on the same fold; they are not checked against the search space.
    :returns: RunResult with predictions, ground truth, and metrics.
    """
    if hyperparameters is not None:
        model_spec = model_spec.with_hyperparameters(hyperparameters, check_values=False)
        hyperparameter_tuning = False
    _, result = fit(
        model_spec,
        dataset,
        fold,
        hyperparameter_tuning=hyperparameter_tuning,
        hpo_metric=hpo_metric,
        hpo_num_trials=hpo_num_trials,
        hpo_random_state=hpo_random_state,
        seed=seed,
        response_transformation=_transformation_for(model_spec, response_transformation),
        wandb_project=wandb_project,
        n_jobs=n_jobs,
    )
    return result


def _transformation_for(
    model: ModelSpec, response_transformation: ResponseTransformer | None
) -> ResponseTransformer | None:
    """Return the transformation *model* trains with: none for the normalization reference."""
    if response_transformation is not None and isinstance(model, ModelSpec) and model.name == NORMALIZATION_REFERENCE:
        logger.info("%s trains on the untransformed response: it is the normalization reference", model.name)
        return None
    return response_transformation
