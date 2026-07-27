"""Cross-study prediction helpers for experiment workflows."""

from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.base import TransformerMixin

from ..datasets.dataset import DrugResponseDataset
from ..models.drp_model import DRPModel
from .training import load_features


def _merge_early_stopping_into_train(
    train_dataset: DrugResponseDataset,
    early_stopping_dataset: DrugResponseDataset | None,
) -> DrugResponseDataset:
    if early_stopping_dataset is not None:
        train_dataset.add_rows(early_stopping_dataset)
    return train_dataset


def _remove_lpo_overlap(train_dataset: DrugResponseDataset, dataset: DrugResponseDataset) -> None:
    train_pairs = {f"{cl}_{drug}" for cl, drug in zip(train_dataset.cell_line_ids, train_dataset.drug_ids, strict=True)}
    dataset_pairs = [f"{cl}_{drug}" for cl, drug in zip(dataset.cell_line_ids, dataset.drug_ids, strict=True)]
    dataset.remove_rows(np.array([i for i, pair in enumerate(dataset_pairs) if pair in train_pairs]))


def _remove_lco_overlap(train_dataset: DrugResponseDataset, dataset: DrugResponseDataset) -> None:
    dataset.reduce_to(
        cell_line_ids=np.setdiff1d(dataset.cell_line_ids, train_dataset.cell_line_ids),
        drug_ids=None,
    )


def _remove_ldo_overlap(train_dataset: DrugResponseDataset, dataset: DrugResponseDataset) -> None:
    dataset.reduce_to(
        cell_line_ids=None,
        drug_ids=np.setdiff1d(dataset.drug_ids, train_dataset.drug_ids),
    )


def _remove_lto_overlap(train_dataset: DrugResponseDataset, dataset: DrugResponseDataset) -> None:
    if train_dataset.tissue is None or dataset.tissue is None:
        raise ValueError("Tissue information not available.")
    train_tissues = set(train_dataset.tissue)
    indices = np.array([i for i, t in enumerate(dataset.tissue) if t not in train_tissues])
    if len(indices) > 0:
        cell_lines_to_keep = np.unique(dataset.cell_line_ids[indices])
    else:
        cell_lines_to_keep = np.array([])
    dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=None)


def remove_train_overlap_for_test_mode(
    test_mode: str,
    train_dataset: DrugResponseDataset,
    dataset: DrugResponseDataset,
) -> None:
    """Remove rows from ``dataset`` that overlap training according to ``test_mode``."""
    if test_mode == "LPO":
        _remove_lpo_overlap(train_dataset, dataset)
    elif test_mode == "LCO":
        _remove_lco_overlap(train_dataset, dataset)
    elif test_mode == "LDO":
        _remove_ldo_overlap(train_dataset, dataset)
    elif test_mode == "LTO":
        _remove_lto_overlap(train_dataset, dataset)
    else:
        raise ValueError(f"Invalid test mode: {test_mode}. Choose from LPO, LCO, LDO, LTO")


def _resolve_drugs_to_keep(
    single_drug_id: str | None,
    drug_features,
) -> np.ndarray | None:
    if single_drug_id is not None:
        return np.array([single_drug_id])
    if drug_features is not None:
        return drug_features.identifiers
    return None


def _predict_cross_study_subset(
    model: DRPModel,
    dataset: DrugResponseDataset,
    cl_features,
    drug_features,
    response_transformation: TransformerMixin | None,
) -> None:
    if len(dataset) == 0:
        dataset._predictions = np.array([])
        return
    drug_input = drug_features.copy() if drug_features is not None else None
    dataset.shuffle(random_state=42)
    dataset._predictions = model.predict(
        cell_line_ids=dataset.cell_line_ids,
        drug_ids=dataset.drug_ids,
        cell_line_input=cl_features.copy(),
        drug_input=drug_input,
    )
    if response_transformation:
        dataset.inverse_transform(response_transformation)


def cross_study_prediction_impl(
    dataset: DrugResponseDataset,
    model: DRPModel,
    test_mode: str,
    train_dataset: DrugResponseDataset,
    path_data: str,
    early_stopping_dataset: DrugResponseDataset | None,
    response_transformation: TransformerMixin | None,
    path_out: str,
    split_index: int,
    single_drug_id: str | None = None,
) -> None:
    """Run cross-study prediction and write CSV output."""
    dataset = dataset.copy()
    os.makedirs(os.path.join(path_out, "cross_study"), exist_ok=True)
    if response_transformation:
        dataset.transform(response_transformation)

    try:
        cl_features, drug_features = load_features(model, path_data, dataset)
    except ValueError as e:
        warnings.warn(str(e), stacklevel=2)
        return

    cell_lines_to_keep = cl_features.identifiers if cl_features is not None else None
    drugs_to_keep = _resolve_drugs_to_keep(single_drug_id, drug_features)

    print(
        f"Reducing cross study dataset ... feature data available for "
        f"{len(cell_lines_to_keep) if cell_lines_to_keep is not None else 'all'} cell lines "
        f"and {len(drugs_to_keep) if drugs_to_keep is not None else 'all'} drugs."
    )

    dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    train_dataset = _merge_early_stopping_into_train(train_dataset, early_stopping_dataset)
    remove_train_overlap_for_test_mode(test_mode, train_dataset, dataset)
    _predict_cross_study_subset(model, dataset, cl_features, drug_features, response_transformation)

    dataset.to_csv(
        os.path.join(
            path_out,
            "cross_study",
            f"cross_study_{dataset.dataset_name}_split_{split_index}.csv",
        )
    )
