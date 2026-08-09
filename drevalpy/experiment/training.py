"""Training and prediction helpers for the experiment path."""

from __future__ import annotations

import numpy as np
from sklearn.base import TransformerMixin
from upath import UPath as Path

from drevalpy.data.structures import SplitMask
from drevalpy.data.structures.dataset import Dataset
from drevalpy.log import get_logger
from drevalpy.models.drp_model import DRPModel
from drevalpy.utils.checkpoints import checkpoint_dir_or_temporary

logger = get_logger(__name__)


def train_and_predict(
    model: DRPModel,
    mudataset: Dataset,
    train_scope: SplitMask,
    test_scope: SplitMask,
    early_stopping_scope: SplitMask | None = None,
    response_transformation: TransformerMixin | None = None,
    model_checkpoint_dir: str | Path | None = None,
) -> np.ndarray:
    """Train the model and predict using Dataset and SplitMask.

    :param model: Untrained DRPModel instance.
    :param mudataset: Full dataset with all features.
    :param train_scope: SplitMask for training samples.
    :param test_scope: SplitMask for test samples.
    :param early_stopping_scope: Optional scope for early stopping.
    :param response_transformation: Optional sklearn response transformer.
    :param model_checkpoint_dir: Directory for checkpoints, or None for temporary.
    :returns: 1-D prediction array for the test set.
    """
    fold_transform = response_transformation

    if fold_transform is not None:
        pairs = train_scope.pairs
        train_cl = pairs[:, 0]
        train_dr = pairs[:, 1]
        response_matrix = mudataset.response_matrix
        train_responses = response_matrix[train_cl, train_dr]
        valid_mask = ~np.isnan(train_responses)
        fold_transform.fit(train_responses[valid_mask].reshape(-1, 1))

    with checkpoint_dir_or_temporary(model_checkpoint_dir) as checkpoint_dir:
        logger.info("Training model ...")
        model.train(
            mudataset=mudataset,
            scope=train_scope,
            early_stopping_scope=early_stopping_scope,
            model_checkpoint_dir=checkpoint_dir,
        )

    logger.info("Predicting ...")
    predictions = model.predict(
        mudataset=mudataset,
        scope=test_scope,
    )

    if fold_transform is not None:
        predictions = fold_transform.inverse_transform(predictions.reshape(-1, 1)).ravel()

    return predictions
