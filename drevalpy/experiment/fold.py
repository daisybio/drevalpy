"""Shared CV-fold preparation for the MuData experiment path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drevalpy.datasets.mudataset import MuDataset
from drevalpy.datasets.splitting import SplitMasks
from drevalpy.models.drp_model import DRPModel


@dataclass(frozen=True)
class MuFoldData:
    """Per-fold data referencing the shared MuDataset with split masks.

    The MuDataset is never copied -- splits are represented as index arrays.
    """

    mudataset: MuDataset
    train_masks: SplitMasks
    val_masks: SplitMasks
    test_masks: SplitMasks
    early_stopping_masks: SplitMasks | None


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

    :returns: MuFoldData with appropriate masks for train/val/test/early_stopping.
    """
    train_masks = SplitMasks(
        train_cell_lines=split_masks.train_cell_lines,
        test_cell_lines=split_masks.train_cell_lines,
        val_cell_lines=np.array([], dtype=np.intp),
        train_drugs=split_masks.train_drugs,
        test_drugs=split_masks.train_drugs,
        val_drugs=np.array([], dtype=np.intp) if split_masks.train_drugs is not None else None,
    )

    val_masks = SplitMasks(
        train_cell_lines=split_masks.val_cell_lines,
        test_cell_lines=split_masks.val_cell_lines,
        val_cell_lines=np.array([], dtype=np.intp),
        train_drugs=split_masks.val_drugs,
        test_drugs=split_masks.val_drugs,
        val_drugs=np.array([], dtype=np.intp) if split_masks.val_drugs is not None else None,
    )

    test_masks = SplitMasks(
        train_cell_lines=split_masks.test_cell_lines,
        test_cell_lines=split_masks.test_cell_lines,
        val_cell_lines=np.array([], dtype=np.intp),
        train_drugs=split_masks.test_drugs,
        test_drugs=split_masks.test_drugs,
        val_drugs=np.array([], dtype=np.intp) if split_masks.test_drugs is not None else None,
    )

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

        val_masks = SplitMasks(
            train_cell_lines=actual_val_cl,
            test_cell_lines=actual_val_cl,
            val_cell_lines=np.array([], dtype=np.intp),
            train_drugs=actual_val_drugs,
            test_drugs=actual_val_drugs,
            val_drugs=np.array([], dtype=np.intp) if actual_val_drugs is not None else None,
        )
        early_stopping_masks = SplitMasks(
            train_cell_lines=es_cl,
            test_cell_lines=es_cl,
            val_cell_lines=np.array([], dtype=np.intp),
            train_drugs=es_drugs,
            test_drugs=es_drugs,
            val_drugs=np.array([], dtype=np.intp) if es_drugs is not None else None,
        )
    else:
        early_stopping_masks = None

    return MuFoldData(
        mudataset=mudataset,
        train_masks=train_masks,
        val_masks=val_masks,
        test_masks=test_masks,
        early_stopping_masks=early_stopping_masks,
    )


def merge_train_val_masks(split_masks: SplitMasks) -> SplitMasks:
    """Merge train and validation masks into a single training set for final training.

    :param split_masks: Original fold split masks.
    :returns: New SplitMasks with train+val merged into train.
    """
    merged_cl = np.concatenate([split_masks.train_cell_lines, split_masks.val_cell_lines])

    merged_drugs: np.ndarray | None = None
    if split_masks.train_drugs is not None and split_masks.val_drugs is not None:
        merged_drugs = np.concatenate([split_masks.train_drugs, split_masks.val_drugs])

    return SplitMasks(
        train_cell_lines=merged_cl,
        test_cell_lines=split_masks.test_cell_lines,
        val_cell_lines=np.array([], dtype=np.intp),
        train_drugs=merged_drugs,
        test_drugs=split_masks.test_drugs,
        val_drugs=None,
    )
