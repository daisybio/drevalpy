"""Shared helpers for split provider tests."""

from __future__ import annotations

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset


def sample_dataset(n_cell_lines: int = 6, n_drugs: int = 4) -> DrugResponseDataset:
    """
    Build a small synthetic response dataset for split tests.

    :param n_cell_lines: number of distinct cell lines
    :param n_drugs: number of distinct drugs per cell line block
    :returns: synthetic ``DrugResponseDataset`` with tissue annotations
    """
    cell_lines = np.repeat([f"CL-{i}" for i in range(n_cell_lines)], n_drugs)
    drugs = np.tile([f"D-{i}" for i in range(n_drugs)], n_cell_lines)
    tissues = np.repeat([f"T-{i % 3}" for i in range(n_cell_lines)], n_drugs)
    return DrugResponseDataset(
        response=np.random.default_rng(0).random(len(cell_lines)),
        cell_line_ids=cell_lines,
        drug_ids=drugs,
        tissues=tissues,
        dataset_name="testset",
    )


def role_from_groups(
    dataset: DrugResponseDataset,
    *,
    train_groups: set[str],
    val_groups: set[str],
    test_groups: set[str],
    group_col: str,
) -> dict[str, DrugResponseDataset]:
    """
    Build one validated split dict from explicit train/validation/test groups.

    :param dataset: source dataset to subset
    :param train_groups: group identifiers assigned to the train role
    :param val_groups: group identifiers assigned to the validation role
    :param test_groups: group identifiers assigned to the test role
    :param group_col: grouping column, one of ``cell_line``, ``drug``, or ``tissue``
    :returns: split dict with ``train``, ``validation``, and ``test`` datasets
    """
    if group_col == "cell_line":
        groups = dataset.cell_line_ids
    elif group_col == "drug":
        groups = dataset.drug_ids
    else:
        assert dataset.tissue is not None
        groups = dataset.tissue

    def subset(selected: set[str]) -> DrugResponseDataset:
        mask = np.isin(groups, list(selected))
        return DrugResponseDataset(
            response=dataset.response[mask],
            cell_line_ids=dataset.cell_line_ids[mask],
            drug_ids=dataset.drug_ids[mask],
            tissues=dataset.tissue[mask] if dataset.tissue is not None else None,
            dataset_name=dataset.dataset_name,
        )

    return {
        "train": subset(train_groups),
        "validation": subset(val_groups),
        "test": subset(test_groups),
    }
