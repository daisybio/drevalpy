"""Shared CV-fold preparation for the MuData experiment path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drevalpy.data.mudataset import MuDataset
from drevalpy.data.splitting import EntityScope, SplitMasks
from drevalpy.models.drp_model import DRPModel


@dataclass(frozen=True)
class MuFoldData:
    """Per-fold data referencing the shared MuDataset with entity scopes.

    The MuDataset is never copied -- splits are represented as index arrays.
    """

    mudataset: MuDataset
    train_scope: EntityScope
    val_scope: EntityScope
    test_scope: EntityScope
    early_stopping_scope: EntityScope | None


def prepare_mu_fold(
    mudataset: MuDataset,
    split_masks: SplitMasks,
    model_class: type[DRPModel],
) -> MuFoldData:
    """Build per-fold MuFoldData from a single SplitMasks.

    For models that support early stopping, the validation set is further
    subdivided into a smaller validation and early-stopping partition.

    :param mudataset: Full dataset (shared across all folds).
    :param split_masks: Fold masks from MuDataSplitter.
    :param model_class: Model class to check for early-stopping support.

    :returns: MuFoldData with appropriate scopes for train/val/test/early_stopping.
    """
    train_scope = EntityScope(
        cell_lines=split_masks.train_cell_lines,
        drugs=split_masks.train_drugs,
    )

    val_scope = EntityScope(
        cell_lines=split_masks.val_cell_lines,
        drugs=split_masks.val_drugs,
    )

    test_scope = EntityScope(
        cell_lines=split_masks.test_cell_lines,
        drugs=split_masks.test_drugs,
    )

    early_stopping_scope: EntityScope | None = None

    if model_class.supports_early_stopping() and len(split_masks.val_cell_lines) > 1:
        n_val = len(split_masks.val_cell_lines)
        n_es = max(1, n_val // 4)
        es_cl = split_masks.val_cell_lines[:n_es]
        actual_val_cl = split_masks.val_cell_lines[n_es:]

        es_drugs = None
        actual_val_drugs = None
        if split_masks.val_drugs is not None:
            es_drugs = split_masks.val_drugs[:n_es]
            actual_val_drugs = split_masks.val_drugs[n_es:]

        val_scope = EntityScope(
            cell_lines=actual_val_cl,
            drugs=actual_val_drugs,
        )
        early_stopping_scope = EntityScope(
            cell_lines=es_cl,
            drugs=es_drugs,
        )

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
    :returns: EntityScope with train+val cell lines/drugs merged.
    """
    merged_cl = np.concatenate([split_masks.train_cell_lines, split_masks.val_cell_lines])

    merged_drugs: np.ndarray | None = None
    if split_masks.train_drugs is not None and split_masks.val_drugs is not None:
        merged_drugs = np.concatenate([split_masks.train_drugs, split_masks.val_drugs])

    return EntityScope(cell_lines=merged_cl, drugs=merged_drugs)
