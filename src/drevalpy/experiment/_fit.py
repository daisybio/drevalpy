"""Tune and train the final model of one execution, then predict its test scope.

:func:`fit` is the one path from an :class:`~drevalpy.experiment._execution.ExecutionSpec` to a
fitted model: ``drevalpy.single``, ``drevalpy.train`` and the golden recorder all go through
it, so they seed, tune, train and predict identically. Tuning is always called: it returns
the spec's values when the execution does not tune.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from drevalpy._utils.seed import fit_seed, seed_everything
from drevalpy.evaluation import evaluate
from drevalpy.experiment._tracking import log_summary, tracking_run
from drevalpy.log import get_logger
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.tuning.hpo import hpam_tune
from drevalpy.types.data.batch.response_batch import ResponseBatch

if TYPE_CHECKING:
    from drevalpy.experiment._execution import ExecutionSpec
    from drevalpy.models.response_transformation import ResponseTransformer
    from drevalpy.types.data.dataset import Dataset
    from drevalpy.types.results.trial import TrialResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class FinalFit:
    """The fitted model of an execution, how it was tuned, and its test-scope predictions."""

    model: DRPModel
    best_hyperparameters: dict[str, Any]
    trials: list[TrialResult]
    #: Test scope, as the pairs in scope order with their observed responses.
    test: ResponseBatch
    #: One prediction per test pair, in scope order.
    predictions: np.ndarray
    metrics: dict[str, float]


def fit(
    execution: ExecutionSpec,
    dataset: Dataset,
    *,
    response_transformation: ResponseTransformer | None = None,
    wandb_project: str | None = None,
    n_jobs: int = 1,
) -> FinalFit:
    """Tune the execution's model, train it on the final training scope and predict the test scope.

    Tuning and the final fit are each seeded from the execution's seed and identity. With
    *wandb_project*, every trial gets a W&B run and the final fit one more, which records the
    test metrics.

    :param execution: Fold decisions: the model, its scopes, tuning settings and seed.
    :param dataset: Full dataset with all features.
    :param response_transformation: Optional unfitted response transformation, cloned for every fit.
    :param wandb_project: W&B project to log to; ``None`` disables tracking.
    :param n_jobs: Worker processes a single-drug predictor fits its per-drug models in.
    :returns: The fitted model, the chosen hyperparameters, the trials and the test predictions.
    """
    identity, scopes = execution.identity, execution.scopes
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
        model.train(dataset, scopes.final_training, early_stopping_scope=scopes.stopping, n_jobs=n_jobs)
        test = ResponseBatch.from_dataset(dataset, scopes.test)
        if len(test.response) == 0:  # ``drevalpy.train`` has no test set
            return FinalFit(model, best, trials, test, np.empty(0), {})
        predictions = model.predict(dataset, scopes.test)
        metrics = evaluate(predictions, test.response)
        log_summary({f"test_{key}": value for key, value in metrics.items()})
    return FinalFit(model, best, trials, test, predictions, metrics)
