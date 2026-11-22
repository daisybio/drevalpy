"""Tune and train the final model of one fold, then predict its test scope.

:func:`fit` prepares the fold and fits the model for ``drevalpy.experiment``, ``drevalpy.run``,
``drevalpy.train`` and the golden recorder. For runs with tuning disabled, the tuner returns the
model recipe's values.

After the final prediction, :func:`fit` checks the non-finite (NaN or infinite) predictions on the test scope, with the
rule written in the "Missing and degenerate input" section of the metrics reference. Tuning
trials are not checked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from drevalpy._utils.seed import fit_seed, seed_everything
from drevalpy.errors import NaNPredictionError
from drevalpy.evaluation import Metric, evaluate
from drevalpy.execution._execution import prepare_fold, result_metadata
from drevalpy.execution._tracking import log_summary, tracking_run
from drevalpy.log import get_logger
from drevalpy.models.config.immutable import freeze_value, thaw_value
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.spec import ModelSpec
from drevalpy.models.tuning.hpo import hpam_tune
from drevalpy.registry.splitter import canonical
from drevalpy.types.data.response_table import ResponseTable
from drevalpy.types.results._identity import RunIdentity
from drevalpy.types.results.run import RunResult

if TYPE_CHECKING:
    from drevalpy.models.response_transformation import ResponseTransformer
    from drevalpy.types.data.dataset import Dataset
    from drevalpy.types.data.fold import Fold

logger = get_logger(__name__)

#: Largest share of non-finite (NaN or infinite) test predictions, among pairs with a finite response, that a run may
#: have and still be scored; a larger share raises :class:`~drevalpy.errors.NaNPredictionError`.
MAX_NAN_PREDICTION_SHARE = 0.2


def fit(
    spec: ModelSpec,
    dataset: Dataset,
    fold: Fold,
    *,
    hyperparameter_tuning: bool,
    hpo_metric: str,
    hpo_num_trials: int,
    hpo_random_state: int | None,
    seed: int,
    response_transformation: ResponseTransformer | None = None,
    wandb_project: str | None = None,
    n_jobs: int = 1,
) -> tuple[DRPModel, RunResult]:
    """Prepare the fold, tune and train the model, and score the test scope.

    Tuning and the final fit are each seeded from the root seed and the run identity. With
    *wandb_project*, every trial gets a W&B run and the final fit one more, which records the
    test metrics.

    :param spec: Named, unfitted model recipe.
    :param dataset: Full dataset with all features.
    :param fold: Original fold before any early-stopping holdout.
    :param hyperparameter_tuning: Whether to search the hyperparameter space.
    :param hpo_metric: Metric to optimize during tuning.
    :param hpo_num_trials: Number of tuning trials.
    :param hpo_random_state: Sampler seed. Defaults to the root seed.
    :param seed: Root seed from which the fit seed is derived.
    :param response_transformation: Optional unfitted response transformation, cloned for every fit.
    :param wandb_project: W&B project to log to; ``None`` disables tracking.
    :param n_jobs: Worker processes a single-drug predictor fits its per-drug models in.
    :returns: The fitted model and the run result: the chosen hyperparameters, the trials and the test-scope
        predictions and metrics, which are empty when the fold has no test scope (``drevalpy.train``).
    :raises TypeError: If *spec* is not a model recipe.
    :raises ValueError: If *hpo_num_trials* is negative.
    """
    if not isinstance(spec, ModelSpec):
        raise TypeError("Experiment models must be ModelSpec values")
    metric = Metric(hpo_metric)
    if hpo_num_trials < 0:
        raise ValueError(f"n_trials must be >= 0 (got {hpo_num_trials}); use 0 for default-only tuning")
    sampler_seed = seed if hpo_random_state is None else hpo_random_state
    mode = canonical(fold.metadata.get("split_mode", ""))
    fold = prepare_fold(spec, dataset, fold, mode)
    metadata = result_metadata(
        spec,
        dataset,
        fold,
        mode,
        hpo={
            "enabled": hyperparameter_tuning,
            "metric": metric.value,
            "num_trials": hpo_num_trials,
            "random_state": sampler_seed,
        },
        seed=seed,
    )
    identity = RunIdentity.from_fold(spec.name, dataset.name, mode, metadata, dataset.randomization)
    metadata = thaw_value(freeze_value(metadata))
    base_config = {
        "model_name": identity.model_name,
        "dataset": dataset.name,
        "split_mode": identity.split_mode,
        "fold_index": identity.fold_index,
        "hyperparameter_tuning": hyperparameter_tuning,
    }
    rng_seed = fit_seed(seed, identity)
    seed_everything(rng_seed)
    best, trials = hpam_tune(
        spec,
        dataset,
        fold,
        tuning=hyperparameter_tuning,
        metric=metric,
        n_trials=hpo_num_trials,
        sampler_seed=sampler_seed,
        response_transformation=response_transformation,
        wandb_project=wandb_project,
        wandb_base_config=base_config,
        n_jobs=n_jobs,
    )
    logger.info("Best hyperparameters: %s", best)
    # best is the spec's stored values or a draw from its space: either was accepted already.
    model = DRPModel(spec, best, check_values=False, response_transformation=response_transformation)
    with tracking_run(
        wandb_project,
        name=f"{identity.model_name}_split{identity.fold_index}_final",
        config={**base_config, **best},
        tags=[identity.model_name, identity.split_mode, dataset.name, "final"],
    ):
        seed_everything(rng_seed)
        training = fold.train_val
        logger.info("Training %s on %d pairs", spec.name, len(training))
        model.train(dataset, training, early_stopping_scope=fold.stopping, n_jobs=n_jobs)
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
        fold_metadata=metadata,
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
