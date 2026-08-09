"""Single model + single fold execution unit."""

from __future__ import annotations

import numpy as np
from sklearn.base import TransformerMixin, clone

from drevalpy.log import get_logger
from drevalpy.models.drp_model import DRPModel
from drevalpy.types import SplitMask, SplitMasks
from drevalpy.types.dataset import Dataset

from .run_result import RunResult
from .trial_result import TrialResult

logger = get_logger(__name__)


def _available_entity_ids(
    mudataset: Dataset,
    views: list[str],
    *,
    side: str,
) -> frozenset[str] | None:
    """Intersect entity availability across all required views.

    Returns None if no views are required (entity_id_only featurizer).
    """
    if not views:
        return None
    available: frozenset[str] | None = None
    for view in views:
        view_entities = mudataset.entities_with_modality(view, side=side)
        available = view_entities if available is None else available & view_entities
    return available


def _filter_to_featurizable_pairs(
    model_class: type[DRPModel],
    mudataset: Dataset,
    split_masks: SplitMasks,
) -> SplitMasks:
    """Filter split masks to only include pairs where both entities have features.

    Zeros out rows (cell lines) or columns (drugs) that lack required modalities.
    """
    config = model_class.model_config()
    cl_views = config.cell_line_views()
    drug_views = config.drug_views()

    available_cl = _available_entity_ids(mudataset, cl_views, side="cell_line")
    available_dr = _available_entity_ids(mudataset, drug_views, side="drug")

    if available_cl is None and available_dr is None:
        return split_masks

    all_cl_ids = mudataset.cell_line_ids
    all_dr_ids = mudataset.drug_ids

    # Build a keepable mask: True for rows/cols with available features
    keep = np.ones(split_masks.shape, dtype=bool)
    if available_cl is not None:
        cl_keep = np.array([cid in available_cl for cid in all_cl_ids])
        keep[~cl_keep, :] = False
    if available_dr is not None:
        dr_keep = np.array([did in available_dr for did in all_dr_ids])
        keep[:, ~dr_keep] = False

    train = split_masks.train & SplitMask(keep)
    test = split_masks.test & SplitMask(keep)
    val = split_masks.val & SplitMask(keep)

    n_before = len(split_masks.train) + len(split_masks.test) + len(split_masks.val)
    n_after = len(train) + len(test) + len(val)
    if n_before != n_after:
        logger.info(
            "Filtered %d pairs to %d featurizable pairs (%.1f%% removed)",
            n_before,
            n_after,
            100.0 * (n_before - n_after) / n_before,
        )

    return SplitMasks(train=train, test=test, val=val, metadata=split_masks.metadata)


def run(
    model_class: type[DRPModel],
    mudataset: Dataset,
    split_masks: SplitMasks,
    *,
    hyperparameter_tuning: bool = True,
    response_transformation: TransformerMixin | None = None,
    hpo_metric: str = "RMSE",
    hpo_num_samples: int = 16,
    hpo_random_state: int = 42,
) -> RunResult:
    """Train a single model on a single fold and predict on the test set.

    :param model_class: DRPModel subclass to train.
    :param mudataset: Full dataset with all features.
    :param split_masks: Single fold's train/test/val boolean masks.
    :param hyperparameter_tuning: Whether to run HPO.
    :param response_transformation: Optional sklearn transformer for responses.
    :param hpo_metric: Metric to optimize during HPO.
    :param hpo_num_samples: Number of HPO trials.
    :param hpo_random_state: Random seed for HPO.
    :returns: RunResult with predictions, ground truth, and metrics.
    """
    from drevalpy.components.core.tuning.config import build_experiment_hpo_config
    from drevalpy.evaluation import AVAILABLE_METRICS
    from drevalpy.utils.checkpoints import checkpoint_dir_or_temporary

    split_masks = _filter_to_featurizable_pairs(model_class, mudataset, split_masks)

    model_name = model_class.get_model_name()
    logger.info("Run: %s, fold %d", model_name, split_masks.metadata.get("fold_index", 0))

    early_stopping_scope: SplitMask | None = None
    val_scope = split_masks.val
    if model_class.supports_early_stopping() and len(split_masks.val) > 1:
        early_stopping_scope, val_scope = split_masks.early_stopping_mask()

    trials: list[TrialResult] | None = None
    if hyperparameter_tuning:
        from drevalpy.components.core.tuning.hpo import hpam_tune_with_trials

        hpo_cfg = build_experiment_hpo_config(
            hpo_metric,
            n_trials=hpo_num_samples,
            random_state=hpo_random_state,
        )
        best_hpams, raw_trials = hpam_tune_with_trials(
            model_class=model_class,
            mudataset=mudataset,
            train_scope=split_masks.train,
            val_scope=val_scope,
            early_stopping_scope=early_stopping_scope,
            response_transformation=response_transformation,
            metric=hpo_metric,
            model_checkpoint_dir=None,
            hpo_config=hpo_cfg,
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
    fold_transform = None if response_transformation is None else clone(response_transformation)

    train_scope = split_masks.train_val
    if fold_transform is not None:
        pairs = train_scope.pairs
        train_responses = mudataset.response_matrix[pairs[:, 0], pairs[:, 1]]
        valid_mask = ~np.isnan(train_responses)
        fold_transform.fit(train_responses[valid_mask].reshape(-1, 1))

    with checkpoint_dir_or_temporary(None) as checkpoint_dir:
        model.train(
            mudataset=mudataset,
            scope=train_scope,
            early_stopping_scope=early_stopping_scope,
            model_checkpoint_dir=checkpoint_dir,
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
        from drevalpy.evaluation import _compute_metric_value

        for metric_name in AVAILABLE_METRICS:
            metrics[metric_name] = _compute_metric_value(metric_name, predictions[valid], ground_truth[valid])

    return RunResult(
        model_name=model_name,
        fold_index=split_masks.metadata.get("fold_index", 0),
        predictions=predictions,
        ground_truth=ground_truth,
        cell_line_ids=cl_ids,
        drug_ids=dr_ids,
        best_hyperparameters=best_hpams,
        metrics=metrics,
        fold_metadata=split_masks.metadata,
        trials=trials,
    )
