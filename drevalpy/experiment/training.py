"""Training and prediction helpers for the MuData experiment path."""

from __future__ import annotations

import numpy as np
from sklearn.base import TransformerMixin
from upath import UPath as Path

from ..datasets.mudataset import MuDataset
from ..datasets.splitting import EntityScope
from ..models.drp_model import DRPModel
from ..utils.checkpoints import checkpoint_dir_or_temporary


def mu_train_and_predict(
    model: DRPModel,
    mudataset: MuDataset,
    train_scope: EntityScope,
    test_scope: EntityScope,
    early_stopping_scope: EntityScope | None = None,
    response_transformation: TransformerMixin | None = None,
    model_checkpoint_dir: str | Path | None = None,
) -> np.ndarray:
    """Train the model and predict using MuDataset and EntityScope.

    No separate feature loading step is needed since MuDataset already
    contains all features.

    :param model: Untrained DRPModel instance.
    :param mudataset: Full dataset with all features.
    :param train_scope: EntityScope for training samples.
    :param test_scope: EntityScope for test samples.
    :param early_stopping_scope: Optional scope for early stopping.
    :param response_transformation: Optional sklearn response transformer.
    :param model_checkpoint_dir: Directory for checkpoints, or None for temporary.

    :returns: 1-D prediction array for the test set.
    """
    fold_transform = response_transformation

    if fold_transform is not None:
        train_cl = train_scope.cell_lines
        train_dr = train_scope.drugs
        response_matrix = mudataset.response_matrix
        if train_dr is not None:
            train_responses = response_matrix[train_cl, train_dr]
        else:
            train_responses = np.nanmean(response_matrix[train_cl, :], axis=1)
        fold_transform.fit(train_responses.reshape(-1, 1))

    with checkpoint_dir_or_temporary(model_checkpoint_dir) as checkpoint_dir:
        print("Training model ...")
        model.train(
            mudataset=mudataset,
            scope=train_scope,
            early_stopping_scope=early_stopping_scope,
            model_checkpoint_dir=checkpoint_dir,
        )

    print("Predicting ...")
    predictions = model.predict(
        mudataset=mudataset,
        scope=test_scope,
    )

    if fold_transform is not None:
        predictions = fold_transform.inverse_transform(predictions.reshape(-1, 1)).ravel()

    return predictions
