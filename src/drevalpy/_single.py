"""Single model + single fold execution unit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from drevalpy._utils.n_jobs import check_n_jobs
from drevalpy.experiment._execution import build_execution
from drevalpy.experiment._fit import fit
from drevalpy.models.spec import ModelSpec
from drevalpy.types import Fold
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.results.run import RunResult

if TYPE_CHECKING:
    from drevalpy.models.response_transformation import ResponseTransformer


def single(
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
    wandb_project: str | None = None,
    n_jobs: int = 1,
) -> RunResult:
    """Train a single model on a single fold and predict on the test set.

    :param model_spec: Named, unfitted model recipe.
    :param dataset: Full dataset with all features.
    :param fold: The fold to train, tune and test on.
    :param hyperparameter_tuning: Whether to run HPO.
    :param response_transformation: Optional unfitted response transformation prototype (any sklearn-compatible
        transformer); a clone is fitted per scope and the caller's instance is left untouched.
    :param hpo_metric: Metric to optimize during HPO.
    :param hpo_num_trials: Number of HPO trials.
    :param hpo_random_state: Random seed for the HPO sampler; defaults to ``seed``.
    :param seed: Root seed; every HPO run and final fit is seeded from it and the execution identity.
    :param wandb_project: Weights & Biases project to log the trials and the final fit to; ``None`` disables
        tracking.
    :param n_jobs: Number of worker processes a single-drug model fits its per-drug models in, during tuning and in the
        final fit; ``-1`` uses all CPUs. The result does not depend on it, and other models ignore it.
    :returns: RunResult with predictions, ground truth, and metrics.
    """
    check_n_jobs(n_jobs)
    return _fit_fold(
        model_spec,
        dataset,
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
) -> RunResult:
    """Run :func:`single` without checking *n_jobs*, for callers that checked it once up front.

    :param model_spec: Named, unfitted model recipe.
    :param dataset: Full dataset with all features.
    :param fold: The fold to train, tune and test on.
    :param hyperparameter_tuning: Whether to run HPO.
    :param response_transformation: Optional unfitted response transformation prototype.
    :param hpo_metric: Metric to optimize during HPO.
    :param hpo_num_trials: Number of HPO trials.
    :param hpo_random_state: Random seed for the HPO sampler; defaults to ``seed``.
    :param seed: Root seed of the HPO runs and the final fit.
    :param wandb_project: Weights & Biases project; ``None`` disables tracking.
    :param n_jobs: Worker processes a single-drug model fits its per-drug models in.
    :returns: RunResult with predictions, ground truth, and metrics.
    """
    execution = build_execution(
        model_spec,
        dataset,
        fold,
        hyperparameter_tuning=hyperparameter_tuning,
        hpo_metric=hpo_metric,
        hpo_num_trials=hpo_num_trials,
        hpo_random_state=seed if hpo_random_state is None else hpo_random_state,
        seed=seed,
    )
    identity = execution.identity
    final = fit(
        execution, dataset, response_transformation=response_transformation, wandb_project=wandb_project, n_jobs=n_jobs
    )
    return RunResult(
        model_name=identity.model_name,
        dataset_name=dataset.name,
        split_mode=identity.split_mode,
        fold_index=identity.fold_index,
        fold_id=identity.fold_id,
        predictions=final.predictions,
        ground_truth=final.test.response,
        sample_ids=final.test.sample_ids,
        drug_ids=final.test.drug_ids,
        best_hyperparameters=final.best_hyperparameters,
        metrics=final.metrics,
        fold_metadata=execution.result_metadata(),
        trials=final.trials or None,
        randomization=dataset.randomization,
    )
