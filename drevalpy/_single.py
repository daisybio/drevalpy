"""Single model + single fold execution unit."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import numpy as np

from drevalpy.evaluation import AVAILABLE_METRICS, _compute_metric_value
from drevalpy.log import get_logger
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.tuning.config import build_experiment_hpo_config
from drevalpy.models.tuning.hpo import hpam_tune
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.data.dataset import Dataset
from drevalpy.types.results.run import RunResult
from drevalpy.types.results.trial import TrialResult
from drevalpy.utils.response_transform import fit_response_transformation

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin

logger = get_logger(__name__)


def single(
    model_class: type[DRPModel],
    mudataset: Dataset,
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

    :param model_class: DRPModel subclass to train.
    :param mudataset: Full dataset with all features.
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
    model_name = model_class.get_model_name()
    logger.info("Run: %s, fold %d", model_name, split_masks.metadata.get("fold_index", 0))

    early_stopping_scope: SplitMask | None = None
    val_scope = split_masks.val
    if model_class.supports_early_stopping() and len(split_masks.val) > 1:
        early_stopping_scope, val_scope = split_masks.early_stopping_mask()

    trials: list[TrialResult] | None = None
    if hyperparameter_tuning:
        hpo_cfg = build_experiment_hpo_config(
            hpo_metric,
            n_trials=hpo_num_samples,
            random_state=hpo_random_state,
        )
        best_hpams, raw_trials = hpam_tune(
            model_class=model_class,
            mudataset=mudataset,
            train_scope=split_masks.train,
            val_scope=val_scope,
            early_stopping_scope=early_stopping_scope,
            response_transformation=response_transformation,
            metric=hpo_metric,
            model_checkpoint_dir=None,
            hpo_config=hpo_cfg,
            precomputed_only=precomputed_only,
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
        best_hpams = model_class.get_default_hyperparameters()

    logger.info("Best hyperparameters: %s", best_hpams)

    model = model_class(best_hpams)

    train_scope = split_masks.train_val
    fold_transform = fit_response_transformation(response_transformation, mudataset, train_scope)

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        model.train(
            mudataset=mudataset,
            scope=train_scope,
            early_stopping_scope=early_stopping_scope,
            model_checkpoint_dir=checkpoint_dir,
            response_transformation=fold_transform,
        )

    predictions = model.predict(mudataset=mudataset, scope=split_masks.test)

    if fold_transform is not None:
        predictions = fold_transform.inverse_transform(predictions.reshape(-1, 1)).ravel()

    response_matrix = mudataset.response_matrix
    test_pairs = split_masks.test.pairs
    ground_truth = response_matrix[test_pairs[:, 0], test_pairs[:, 1]]

    cl_ids = mudataset.cell_line_ids[test_pairs[:, 0]]
    dr_ids = mudataset.drug_ids[test_pairs[:, 1]]

    valid = ~np.isnan(predictions) & ~np.isnan(ground_truth)
    metrics: dict[str, float] = {}
    if valid.any():
        for metric_name in AVAILABLE_METRICS:
            metrics[metric_name] = _compute_metric_value(metric_name, predictions[valid], ground_truth[valid])

    return RunResult(
        model_name=model_name,
        dataset_name=mudataset.name,
        split_mode=split_masks.metadata.get("split_mode", ""),
        fold_index=split_masks.metadata.get("fold_index", 0),
        fold_id=split_masks.metadata.get("fold_id", ""),
        predictions=predictions,
        ground_truth=ground_truth,
        cell_line_ids=cl_ids,
        drug_ids=dr_ids,
        best_hyperparameters=best_hpams,
        metrics=metrics,
        # Copied: the caller's SplitMasks.metadata is reused across models and
        # folds by run(), so the result must not alias it.
        fold_metadata=dict(split_masks.metadata),
        trials=trials,
        randomization=mudataset.randomization,
    )
