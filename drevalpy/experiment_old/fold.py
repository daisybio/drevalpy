"""Fold preparation for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drevalpy.data.structures import EntityScope, SplitMasks
from drevalpy.data.structures.dataset import Dataset
from drevalpy.models.drp_model import DRPModel


@dataclass(frozen=True)
class MuFoldData:
    """All data for one CV fold.

    The Dataset is never copied -- splits are represented as pair arrays.
    """

    mudataset: Dataset
    train_scope: EntityScope
    val_scope: EntityScope
    test_scope: EntityScope
    early_stopping_scope: EntityScope | None


def prepare_mu_fold(
    mudataset: Dataset,
    split_masks: SplitMasks,
    model_class: type[DRPModel],
) -> MuFoldData:
    """Build fold data from SplitMasks.

    :param mudataset: Full dataset (not copied).
    :param split_masks: Fold masks with 2D pair arrays.
    :param model_class: Model class to check for early-stopping support.

    :returns: MuFoldData with appropriate scopes for train/val/test/early_stopping.
    """
    train_scope = EntityScope(pairs=split_masks.train)
    val_pairs = split_masks.val
    test_scope = EntityScope(pairs=split_masks.test)

    early_stopping_scope: EntityScope | None = None

    if model_class.supports_early_stopping() and len(val_pairs) > 1:
        n_val = len(val_pairs)
        n_es = max(1, n_val // 4)
        early_stopping_scope = EntityScope(pairs=val_pairs[:n_es])
        val_pairs = val_pairs[n_es:]

    val_scope = EntityScope(pairs=val_pairs)

    return MuFoldData(
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_scope,
        test_scope=test_scope,
        early_stopping_scope=early_stopping_scope,
    )


def merge_train_val_scopes(split_masks: SplitMasks) -> EntityScope:
    """Merge train and validation into a single training EntityScope.

    :param split_masks: Original fold split masks.
    :returns: EntityScope with train+val pairs merged.
    """
    merged = np.concatenate([split_masks.train, split_masks.val])
    return EntityScope(pairs=merged)
