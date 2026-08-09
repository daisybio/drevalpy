"""Final production model training for experiment workflows."""

from __future__ import annotations

from sklearn.base import TransformerMixin
from upath import UPath as Path

from drevalpy.log import get_logger

from ..data.structures.dataset import Dataset
from ..models.drp_model import DRPModel
from ..utils.checkpoints import checkpoint_dir_or_temporary
from .fold import merge_train_val_scopes, prepare_mu_fold
from .hpo import select_final_model_hyperparameters

logger = get_logger(__name__)


def train_final_model_impl(
    model_class: type[DRPModel],
    mudataset: Dataset,
    response_transformation: TransformerMixin | None,
    model_checkpoint_dir: str | Path | None,
    metric: str,
    final_model_path: str | Path,
    test_mode: str = "LCO",
    val_ratio: float = 0.1,
    hyperparameter_tuning: bool = True,
    hpo_num_samples: int = 16,
    hpo_random_state: int = 42,
) -> None:
    """Train and persist a final production model on the full dataset.

    :param model_class: Model class to train.
    :param mudataset: Full Dataset for final training.
    :param response_transformation: Response transformer fitted on training data.
    :param model_checkpoint_dir: Directory for intermediate checkpoints, or ``None`` for a temporary one.
    :param metric: Metric optimized during optional hyperparameter tuning.
    :param final_model_path: Archive path stem for the final model (``.zip`` appended on save).
    :param test_mode: Split mode for the internal train/validation holdout.
    :param val_ratio: Validation fraction for the holdout split.
    :param hyperparameter_tuning: Whether to tune hyperparameters before training.
    :param hpo_num_samples: Number of HPO trials when tuning is enabled.
    :param hpo_random_state: Random seed for hyperparameter search.
    """
    from drevalpy.components.core.tuning.config import build_experiment_hpo_config
    from drevalpy.data.splitters import splitter_registry

    logger.info("Training final model with application-specific validation strategy ...")

    splitter = splitter_registry.get(test_mode)
    folds = splitter(
        mudataset,
        n_splits=5,
        validation_ratio=val_ratio,
        random_state=hpo_random_state,
    )
    split_masks = folds[0]
    fold_data = prepare_mu_fold(mudataset, split_masks, model_class)

    hpo_cfg = build_experiment_hpo_config(
        metric,
        n_trials=hpo_num_samples,
        random_state=hpo_random_state,
    )
    best_hpams = select_final_model_hyperparameters(
        model_class=model_class,
        mudataset=mudataset,
        train_scope=fold_data.train_scope,
        val_scope=fold_data.val_scope,
        early_stopping_scope=fold_data.early_stopping_scope,
        response_transformation=response_transformation,
        metric=metric,
        model_checkpoint_dir=model_checkpoint_dir,
        hyperparameter_tuning=hyperparameter_tuning,
        hpo_config=hpo_cfg,
    )

    logger.info("Best hyperparameters for final model: %s", best_hpams)
    model = model_class(best_hpams)

    merged_scope = merge_train_val_scopes(split_masks)

    with checkpoint_dir_or_temporary(model_checkpoint_dir) as checkpoint_dir:
        model.train(
            mudataset=mudataset,
            scope=merged_scope,
            model_checkpoint_dir=checkpoint_dir,
        )

    final_model_target = Path(final_model_path)
    final_model_target.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_model_target)
