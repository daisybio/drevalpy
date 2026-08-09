"""Fold preparation for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drevalpy.data.structures import SplitMask, SplitMasks
from drevalpy.data.structures.dataset import Dataset
from drevalpy.models.drp_model import DRPModel


@dataclass(frozen=True)
class MuFoldData:
    """All data for one CV fold.

    The Dataset is never copied -- splits are represented as boolean masks.
    """

    mudataset: Dataset
    train_scope: SplitMask
    val_scope: SplitMask
    test_scope: SplitMask
    early_stopping_scope: SplitMask | None


def prepare_mu_fold(
    mudataset: Dataset,
    split_masks: SplitMasks,
    model_class: type[DRPModel],
) -> MuFoldData:
    """Build fold data from SplitMasks.

    :param mudataset: Full dataset (not copied).
    :param split_masks: Fold masks with 2D boolean arrays.
    :param model_class: Model class to check for early-stopping support.

    :returns: MuFoldData with appropriate scopes for train/val/test/early_stopping.
    """
    train_scope = split_masks.train
    val_mask = split_masks.val
    test_scope = split_masks.test

    early_stopping_scope: SplitMask | None = None

    n_val = len(val_mask)
    if model_class.supports_early_stopping() and n_val > 1:
        val_pairs = val_mask.pairs
        n_es = max(1, n_val // 4)
        es_pairs = val_pairs[:n_es]
        remaining_pairs = val_pairs[n_es:]

        es_arr = np.zeros_like(val_mask.mask)
        es_arr[es_pairs[:, 0], es_pairs[:, 1]] = True
        early_stopping_scope = SplitMask(es_arr)

        val_arr = np.zeros_like(val_mask.mask)
        val_arr[remaining_pairs[:, 0], remaining_pairs[:, 1]] = True
        val_mask = SplitMask(val_arr)

    return MuFoldData(
        mudataset=mudataset,
        train_scope=train_scope,
        val_scope=val_mask,
        test_scope=test_scope,
        early_stopping_scope=early_stopping_scope,
    )


def merge_train_val_scopes(split_masks: SplitMasks) -> SplitMask:
    """Merge train and validation into a single training SplitMask.

    :param split_masks: Original fold split masks.
    :returns: SplitMask with train | val masks merged.
    """
    return split_masks.train | split_masks.val
