"""Single model + single fold execution unit."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

from drevalpy.evaluation import score_predictions
from drevalpy.experiment._execution import ExecutionSpec, build_execution
from drevalpy.log import get_logger
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.spec import ModelSpec
from drevalpy.models.tuning.config import build_experiment_hpo_config
from drevalpy.models.tuning.hpo import hpam_tune
from drevalpy.types import SplitMasks
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.results.run import RunResult
from drevalpy.types.results.trial import TrialResult

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin

logger = get_logger(__name__)


def single(
    model_spec: ModelSpec,
    dataset: Dataset,
    split_masks: SplitMasks,
    *,
    hyperparameter_tuning: bool = True,
    response_transformation: TransformerMixin | None = None,
    hpo_metric: str = "RMSE",
    hpo_num_samples: int = 16,
    hpo_random_state: int = 42,
    precomputed_only: bool = False,
) -> RunResult:
    """Train a single model on a single fold and predict on the test set.

    :param model_spec: Named, unfitted model recipe.
    :param dataset: Full dataset with all features.
    :param split_masks: Single fold's train/test/val boolean masks.
    :param hyperparameter_tuning: Whether to run HPO.
    :param response_transformation: Optional unfitted sklearn transformer prototype; a clone is
        fitted per scope and the caller's instance is left untouched.
    :param hpo_metric: Metric to optimize during HPO.
    :param hpo_num_samples: Number of HPO trials.
    :param hpo_random_state: Random seed for HPO.
    :param precomputed_only: Restrict HPO to pre-computed featurizer variants.
    :returns: RunResult with predictions, ground truth, and metrics.
    """
    execution = build_execution(
        model_spec,
        dataset,
        split_masks,
        hyperparameter_tuning=hyperparameter_tuning,
        hpo_metric=hpo_metric,
        hpo_num_samples=hpo_num_samples,
        hpo_random_state=hpo_random_state,
        precomputed_only=precomputed_only,
    )
    return _execute(execution, dataset, response_transformation)


def _execute(execution: ExecutionSpec, dataset: Dataset, response_transformation=None) -> RunResult:
    """Execute established scopes; model pipelines own all fitted transforms."""
    model_spec, scopes, identity = execution.model, execution.scopes, execution.identity
    hpo_metric = execution.hpo_metric
    trials: list[TrialResult] | None = None
    if execution.hyperparameter_tuning:
        hpo_cfg = build_experiment_hpo_config(
            hpo_metric,
            n_trials=execution.hpo_num_samples,
            random_state=execution.hpo_random_state,
        )
        best_hpams, raw_trials = hpam_tune(
            model_spec=model_spec,
            mudataset=dataset,
            train_scope=scopes.training,
            val_scope=scopes.validation,
            early_stopping_scope=scopes.stopping,
            response_transformation=response_transformation,
            metric=hpo_metric,
            model_checkpoint_dir=None,
            hpo_config=hpo_cfg,
            precomputed_only=execution.precomputed_only,
        )
        trials = [
            TrialResult(
                hyperparameters=params,
                metrics=trial_metrics,
                optimization_metric=hpo_metric,
                predictions=preds,
            )
            for params, trial_metrics, preds in raw_trials
        ]
    else:
        best_hpams = model_spec.default_hyperparameters

    logger.info("Best hyperparameters: %s", best_hpams)

    model = DRPModel(model_spec, best_hpams, response_transformation=response_transformation)

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.train(
            dataset=dataset,
            scope=scopes.final_training,
            early_stopping_scope=scopes.stopping,
            model_checkpoint_dir=checkpoint_dir,
        )

    predictions = model.predict(dataset=dataset, scope=scopes.test)

    test_response = ResponseBatch.from_dataset(dataset, scopes.test)
    ground_truth = test_response.response
    metrics = score_predictions(predictions, ground_truth, empty_as_dict=True)
    return RunResult(
        model_name=identity.model_name,
        dataset_name=dataset.name,
        split_mode=identity.split_mode,
        fold_index=identity.fold_index,
        fold_id=identity.fold_id,
        predictions=predictions,
        ground_truth=ground_truth,
        cell_line_ids=test_response.cell_line_ids,
        drug_ids=test_response.drug_ids,
        best_hyperparameters=best_hpams,
        metrics=metrics,
        fold_metadata=execution.result_metadata(),
        trials=trials,
        randomization=dataset.randomization,
    )
