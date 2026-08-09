"""Robustness testing helpers for experiment workflows.

Tests model stability by re-training with shuffled index orderings
(different random seeds) and comparing prediction consistency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, clone
from upath import UPath as Path

from drevalpy.log import get_logger

from ..data.structures import EntityScope
from ..data.structures.mudataset import MuDataset
from ..models.drp_model import DRPModel
from .training import mu_train_and_predict

logger = get_logger(__name__)


def _shuffle_scope(scope: EntityScope, rng: np.random.Generator) -> EntityScope:
    """Return a copy of scope with pair array shuffled."""
    shuffled_pairs = rng.permutation(scope.pairs)
    return EntityScope(pairs=shuffled_pairs)


def _write_robustness_predictions(
    prediction_file: Path,
    mudataset: MuDataset,
    test_scope: EntityScope,
    predictions: np.ndarray,
) -> None:
    """Write robustness trial prediction CSV."""
    cl_ids = mudataset.cell_line_ids
    drug_ids = mudataset.drug_ids
    response_matrix = mudataset.response_matrix

    pairs = test_scope.pairs
    cl_idx = pairs[:, 0]
    dr_idx = pairs[:, 1]

    rows: dict[str, Any] = {
        "cell_line_ids": cl_ids[cl_idx],
        "drug_ids": drug_ids[dr_idx],
        "response": response_matrix[cl_idx, dr_idx],
        "predictions": predictions,
    }

    df = pd.DataFrame(rows)
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(prediction_file, index=False)


def robustness_train_predict_impl(
    trial: int,
    trial_file: str | Path,
    mudataset: MuDataset,
    train_scope: EntityScope,
    test_scope: EntityScope,
    early_stopping_scope: EntityScope | None,
    model_class: type[DRPModel],
    hyperparameters: dict[str, Any],
    response_transformation: TransformerMixin | None = None,
    model_checkpoint_dir: str | Path | None = None,
) -> None:
    """Train and predict for one robustness trial.

    :param trial: Trial index (used as random seed for shuffling).
    :param trial_file: Output path for predictions.
    :param mudataset: Full MuDataset with all features.
    :param train_scope: EntityScope for training samples.
    :param test_scope: EntityScope for test samples.
    :param early_stopping_scope: Optional EntityScope for early stopping.
    :param model_class: Model class to train on perturbed data.
    :param hyperparameters: Hyperparameters for model construction.
    :param response_transformation: Optional response transformer.
    :param model_checkpoint_dir: Directory for model checkpoints, or ``None`` for a temporary one.
    """
    rng = np.random.default_rng(trial)
    shuffled_train = _shuffle_scope(train_scope, rng)
    shuffled_test = _shuffle_scope(test_scope, rng)
    shuffled_es = _shuffle_scope(early_stopping_scope, rng) if early_stopping_scope is not None else None

    trial_model = model_class(hyperparameters)
    trial_transform = None if response_transformation is None else clone(response_transformation)

    predictions = mu_train_and_predict(
        model=trial_model,
        mudataset=mudataset,
        train_scope=shuffled_train,
        test_scope=shuffled_test,
        early_stopping_scope=shuffled_es,
        response_transformation=trial_transform,
        model_checkpoint_dir=model_checkpoint_dir,
    )

    _write_robustness_predictions(Path(trial_file), mudataset, shuffled_test, predictions)


def robustness_test_impl(
    n_trials: int,
    model_class: type[DRPModel],
    hyperparameters: dict[str, Any],
    mudataset: MuDataset,
    train_scope: EntityScope,
    test_scope: EntityScope,
    early_stopping_scope: EntityScope | None,
    path_out: str | Path,
    split_index: int,
    response_transformation: TransformerMixin | None = None,
    model_checkpoint_dir: str | Path | None = None,
) -> None:
    """Run robustness tests with varying shuffle seeds.

    :param n_trials: Number of robustness trials to run.
    :param model_class: Model class to retrain on perturbed data.
    :param hyperparameters: Hyperparameters for model construction.
    :param mudataset: Full MuDataset with all features.
    :param train_scope: EntityScope for training samples.
    :param test_scope: EntityScope for test samples.
    :param early_stopping_scope: Optional EntityScope for early stopping.
    :param path_out: Directory where predictions are written.
    :param split_index: CV fold index for output file naming.
    :param response_transformation: Optional response transformer.
    :param model_checkpoint_dir: Directory for model checkpoints, or ``None`` for a temporary one.
    """
    robustness_test_path = Path(path_out) / "robustness"
    robustness_test_path.mkdir(parents=True, exist_ok=True)
    for trial in range(n_trials):
        logger.info("Running robustness test trial %d/%d", trial + 1, n_trials)
        trial_file = robustness_test_path / f"robustness_{trial + 1}_split_{split_index}.csv"
        if trial_file.is_file():
            continue
        robustness_train_predict_impl(
            trial=trial,
            trial_file=trial_file,
            mudataset=mudataset,
            train_scope=train_scope,
            test_scope=test_scope,
            early_stopping_scope=early_stopping_scope,
            model_class=model_class,
            hyperparameters=hyperparameters,
            response_transformation=response_transformation,
            model_checkpoint_dir=model_checkpoint_dir,
        )
