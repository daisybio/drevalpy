"""Cross-study prediction helpers for experiment workflows.

Evaluates a model trained on one study against a held-out target study,
removing overlap according to the test mode (LPO, LCO, LDO, LTO).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from upath import UPath as Path

from ..data.structures import EntityScope, SplitMasks
from ..data.structures.mudataset import MuDataset
from ..models.drp_model import DRPModel


def _all_pairs_indices(mudataset: MuDataset) -> tuple[np.ndarray, np.ndarray]:
    """Return all non-NaN (row, col) index pairs from the response matrix."""
    response = mudataset.response_matrix
    return np.where(~np.isnan(response))


def _remove_lpo_overlap(
    train_masks: SplitMasks,
    source: MuDataset,
    target_cl_idx: np.ndarray,
    target_dr_idx: np.ndarray,
    target: MuDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove pairs that appear in both training and target (leave-pair-out)."""
    src_cl_ids = source.cell_line_ids
    src_drug_ids = source.drug_ids
    train_pairs_arr = train_masks.train
    train_cl = train_pairs_arr[:, 0]
    train_dr = train_pairs_arr[:, 1]

    train_pairs = {f"{src_cl_ids[c]}_{src_drug_ids[d]}" for c, d in zip(train_cl, train_dr, strict=True)}

    tgt_cl_ids = target.cell_line_ids
    tgt_drug_ids = target.drug_ids
    keep = np.array(
        [
            i
            for i in range(len(target_cl_idx))
            if f"{tgt_cl_ids[target_cl_idx[i]]}_{tgt_drug_ids[target_dr_idx[i]]}" not in train_pairs
        ],
        dtype=np.intp,
    )
    return target_cl_idx[keep], target_dr_idx[keep]


def _remove_lco_overlap(
    train_masks: SplitMasks,
    source: MuDataset,
    target_cl_idx: np.ndarray,
    target_dr_idx: np.ndarray,
    target: MuDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove cell lines that appear in training (leave-cell-line-out)."""
    train_cl_indices = train_masks.train[:, 0]
    train_cl_names = set(source.cell_line_ids[train_cl_indices])
    tgt_cl_ids = target.cell_line_ids
    keep = np.array(
        [i for i in range(len(target_cl_idx)) if tgt_cl_ids[target_cl_idx[i]] not in train_cl_names],
        dtype=np.intp,
    )
    return target_cl_idx[keep], target_dr_idx[keep]


def _remove_ldo_overlap(
    train_masks: SplitMasks,
    source: MuDataset,
    target_cl_idx: np.ndarray,
    target_dr_idx: np.ndarray,
    target: MuDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove drugs that appear in training (leave-drug-out)."""
    train_dr_indices = train_masks.train[:, 1]
    train_drug_names = set(source.drug_ids[train_dr_indices])
    tgt_drug_ids = target.drug_ids
    keep = np.array(
        [i for i in range(len(target_cl_idx)) if tgt_drug_ids[target_dr_idx[i]] not in train_drug_names],
        dtype=np.intp,
    )
    return target_cl_idx[keep], target_dr_idx[keep]


def _remove_lto_overlap(
    train_masks: SplitMasks,
    source: MuDataset,
    target_cl_idx: np.ndarray,
    target_dr_idx: np.ndarray,
    target: MuDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove tissues that appear in training (leave-tissue-out)."""
    train_cl_ids = source.cell_line_ids[train_masks.train[:, 0]]
    train_tissues = set(source.get_tissue(train_cl_ids))
    tgt_cl_ids = target.cell_line_ids
    tgt_tissues = target.get_tissue(tgt_cl_ids)
    keep = np.array(
        [i for i in range(len(target_cl_idx)) if tgt_tissues[target_cl_idx[i]] not in train_tissues],
        dtype=np.intp,
    )
    return target_cl_idx[keep], target_dr_idx[keep]


def _remove_train_overlap(
    test_mode: str,
    train_masks: SplitMasks,
    source: MuDataset,
    target_cl_idx: np.ndarray,
    target_dr_idx: np.ndarray,
    target: MuDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the appropriate overlap-removal function."""
    dispatch = {
        "LPO": _remove_lpo_overlap,
        "LCO": _remove_lco_overlap,
        "LDO": _remove_ldo_overlap,
        "LTO": _remove_lto_overlap,
    }
    if test_mode not in dispatch:
        raise ValueError(f"Invalid test mode: {test_mode}. Choose from LPO, LCO, LDO, LTO")
    return dispatch[test_mode](train_masks, source, target_cl_idx, target_dr_idx, target)


def _write_cross_study_predictions(
    prediction_file: Path,
    target: MuDataset,
    test_cl_idx: np.ndarray,
    test_dr_idx: np.ndarray,
    predictions: np.ndarray,
) -> None:
    """Write cross-study prediction CSV."""
    cl_ids = target.cell_line_ids
    drug_ids = target.drug_ids
    response_matrix = target.response_matrix

    df = pd.DataFrame(
        {
            "cell_line_ids": cl_ids[test_cl_idx],
            "drug_ids": drug_ids[test_dr_idx],
            "predictions": predictions,
            "response": response_matrix[test_cl_idx, test_dr_idx],
        }
    )
    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(prediction_file, index=False)


def cross_study_prediction_impl(
    target: MuDataset,
    model: DRPModel,
    test_mode: str,
    train_masks: SplitMasks,
    source: MuDataset,
    path_out: str | Path,
    split_index: int,
    dataset_name: str = "cross_study",
) -> None:
    """Run cross-study prediction and write CSV output.

    :param target: Held-out MuDataset from another study.
    :param model: Already-trained model instance.
    :param test_mode: Split mode for overlap removal (LPO, LCO, LDO, LTO).
    :param train_masks: SplitMasks used when training the model on the source study.
    :param source: Source MuDataset the model was trained on.
    :param path_out: Directory where predictions are written.
    :param split_index: CV fold index for output file naming.
    :param dataset_name: Name for the target dataset (used in the output filename).
    """
    target_cl_idx, target_dr_idx = _all_pairs_indices(target)

    target_cl_idx, target_dr_idx = _remove_train_overlap(
        test_mode, train_masks, source, target_cl_idx, target_dr_idx, target
    )

    if len(target_cl_idx) == 0:
        warnings.warn(
            f"No samples remaining after overlap removal for cross-study dataset {dataset_name}.",
            stacklevel=2,
        )
        return

    print(f"Cross-study prediction: {len(target_cl_idx)} samples after overlap removal.")

    test_pairs = np.column_stack([target_cl_idx, target_dr_idx])
    test_scope = EntityScope(pairs=test_pairs)
    predictions = model.predict(mudataset=target, scope=test_scope)

    output_dir = Path(path_out) / "cross_study"
    output_file = output_dir / f"cross_study_{dataset_name}_split_{split_index}.csv"
    _write_cross_study_predictions(output_file, target, target_cl_idx, target_dr_idx, predictions)
