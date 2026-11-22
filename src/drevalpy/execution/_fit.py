"""Tune and train the final model of one execution, then predict its test scope.

:func:`fit` is the one path from an :class:`~drevalpy.execution._execution.ExecutionSpec` to a
fitted model: ``drevalpy.run``, ``drevalpy.train`` and the golden recorder all go through
it, so they seed, tune, train and predict identically. Tuning is always called: it returns
the spec's values when the execution does not tune.

After the final prediction, :func:`fit` checks the non-finite (NaN or infinite) predictions on the test scope, with the
rule written in the "Missing and degenerate input" section of the metrics reference. Tuning
trials are not checked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy._utils.seed import fit_seed, seed_everything
from drevalpy.errors import NaNPredictionError
from drevalpy.evaluation import evaluate
from drevalpy.execution._tracking import log_summary, tracking_run
from drevalpy.log import get_logger
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.tuning.hpo import hpam_tune
from drevalpy.types.data.response_table import ResponseTable
from drevalpy.types.results.run import RunResult

if TYPE_CHECKING:
    from drevalpy.execution._execution import ExecutionSpec
    from drevalpy.models.response_transformation import ResponseTransformer
    from drevalpy.types.data.dataset import Dataset

logger = get_logger(__name__)

#: Largest share of non-finite (NaN or infinite) test predictions, among pairs with a finite response, that a run may
#: have and still be scored; a larger share raises :class:`~drevalpy.errors.NaNPredictionError`.
MAX_NAN_PREDICTION_SHARE = 0.2


def fit(
    execution: ExecutionSpec,
    dataset: Dataset,
    *,
    response_transformation: ResponseTransformer | None = None,
    wandb_project: str | None = None,
    n_jobs: int = 1,
) -> tuple[DRPModel, RunResult]:
    """Tune the execution's model, train it on the final training scope and predict the test scope.

    Tuning and the final fit are each seeded from the execution's seed and identity. With
    *wandb_project*, every trial gets a W&B run and the final fit one more, which records the
    test metrics.

    :param execution: Fold decisions: the model, its scopes, tuning settings and seed.
    :param dataset: Full dataset with all features.
    :param response_transformation: Optional unfitted response transformation, cloned for every fit.
    :param wandb_project: W&B project to log to; ``None`` disables tracking.
    :param n_jobs: Worker processes a single-drug predictor fits its per-drug models in.
    :returns: The fitted model and the run result: the chosen hyperparameters, the trials and the test-scope
        predictions and metrics, which are empty when the execution has no test scope (``drevalpy.train``).
    """
    identity, fold = execution.identity, execution.fold
    base_config = {
        "model_name": identity.model_name,
        "dataset": dataset.name,
        "split_mode": identity.split_mode,
        "fold_index": identity.fold_index,
        "hyperparameter_tuning": execution.hyperparameter_tuning,
    }
    rng_seed = fit_seed(execution.seed, identity)
    seed_everything(rng_seed)
    best, trials = hpam_tune(
        execution,
        dataset,
        response_transformation=response_transformation,
        wandb_project=wandb_project,
        wandb_base_config=base_config,
        n_jobs=n_jobs,
    )
    logger.info("Best hyperparameters: %s", best)
    # best is the spec's stored values or a draw from its space: either was accepted already.
    model = DRPModel(execution.model, best, check_values=False, response_transformation=response_transformation)
    with tracking_run(
        wandb_project,
        name=f"{identity.model_name}_split{identity.fold_index}_final",
        config={**base_config, **best},
        tags=[identity.model_name, identity.split_mode, dataset.name, "final"],
    ):
        seed_everything(rng_seed)
        model.train(dataset, fold.train_val, early_stopping_scope=fold.stopping, n_jobs=n_jobs)
        test = ResponseTable.from_dataset(dataset, fold.test)
        if len(test.response) == 0:  # ``drevalpy.train`` has no test set
            predictions, metrics = np.empty(0), {}
        else:
            predictions = model.predict(dataset, fold.test)
            _check_non_finite_predictions(identity.model_name, predictions, test.response)
            metrics = evaluate(predictions, test.response)
            log_summary({f"test_{key}": value for key, value in metrics.items()})
    result = RunResult(
        model_name=identity.model_name,
        dataset_name=dataset.name,
        split_mode=identity.split_mode,
        fold_index=identity.fold_index,
        fold_id=identity.fold_id,
        predictions=predictions,
        ground_truth=test.response,
        sample_ids=test.sample_ids,
        drug_ids=test.drug_ids,
        best_hyperparameters=best,
        metrics=metrics,
        fold_metadata=execution.result_metadata(),
        trials=trials or None,
        randomization=dataset.randomization,
    )
    return model, result


def _check_non_finite_predictions(model_name: str, predictions: np.ndarray, ground_truth: np.ndarray) -> None:
    """Warn about non-finite test predictions and refuse to score a run with too many of them.

    Only pairs with a finite response count, since scoring drops the others anyway.

    :param model_name: Model name, for the messages.
    :param predictions: One prediction per test pair.
    :param ground_truth: The observed response of each test pair.
    :raises NaNPredictionError: If more than :data:`MAX_NAN_PREDICTION_SHARE` of the pairs with a
        finite response have a NaN or infinite prediction.
    """
    scored = np.isfinite(ground_truth)
    n_pairs = int(scored.sum())
    n_bad = int((~np.isfinite(predictions[scored])).sum())
    if n_bad == 0:
        return
    logger.warning(
        "%s: %d of %d test predictions are NaN or infinite and are excluded from scoring", model_name, n_bad, n_pairs
    )
    if n_bad > MAX_NAN_PREDICTION_SHARE * n_pairs:
        raise NaNPredictionError(
            f"{model_name}: {n_bad} of {n_pairs} test predictions are NaN or infinite, more than "
            f"{MAX_NAN_PREDICTION_SHARE:.0%}; the model cannot predict these pairs, for example because "
            "their features are missing or it has no state for their sample or drug."
        )
