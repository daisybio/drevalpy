"""Example custom split script for LCO-style scaling-law experiments.

Define train/validation/test subsets in this file. ``test_mode=LCO`` must be used when
running drevalpy so cell-line disjointness is validated on the produced splits.

Custom split scripts execute as local Python code. drevalpy validates obvious
overlap/leakage for the selected ``test_mode``, but cannot guarantee that the split
answers your scientific question.
"""

from __future__ import annotations

import numpy as np

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.datasets.splits import SplitParams

# Fraction of cell lines held out for test; remaining cell lines are split train/val.
TEST_FRACTION = 0.2


def _subset(dataset: DrugResponseDataset, mask: np.ndarray) -> DrugResponseDataset:
    return DrugResponseDataset(
        response=dataset.response[mask],
        cell_line_ids=dataset.cell_line_ids[mask],
        drug_ids=dataset.drug_ids[mask],
        tissues=dataset.tissue[mask] if dataset.tissue is not None else None,
        dataset_name=dataset.dataset_name,
    )


def create_splits(
    response_data: DrugResponseDataset,
    params: SplitParams,
) -> list[dict[str, DrugResponseDataset]]:
    """
    Return one LCO-style split with configurable train/validation/test cell-line groups.

    :param response_data: full response dataset to partition
    :param params: pipeline split settings (seed, validation ratio, fold count, etc.)
    :returns: list containing one split dict with train, validation, and test roles
    """
    rng = np.random.default_rng(params.random_state)
    unique_cell_lines = np.unique(response_data.cell_line_ids)
    shuffled = rng.permutation(unique_cell_lines)

    n_test = max(1, int(len(shuffled) * TEST_FRACTION))
    n_val = max(1, int(len(shuffled) * params.validation_ratio))
    n_val = min(n_val, len(shuffled) - n_test - 1)

    test_cls = set(shuffled[:n_test])
    val_cls = set(shuffled[n_test : n_test + n_val])  # noqa: E203
    train_cls = set(shuffled[n_test + n_val :])  # noqa: E203

    train_mask = np.isin(response_data.cell_line_ids, list(train_cls))
    val_mask = np.isin(response_data.cell_line_ids, list(val_cls))
    test_mask = np.isin(response_data.cell_line_ids, list(test_cls))

    split = {
        "train": _subset(response_data, train_mask),
        "validation": _subset(response_data, val_mask),
        "test": _subset(response_data, test_mask),
        "metadata": {
            "fraction_test": TEST_FRACTION,
            "fraction_validation": params.validation_ratio,
            "seed": params.random_state,
            "n_cv_splits": params.n_cv_splits,
            "test_mode": params.test_mode,
        },
    }
    return [split]
