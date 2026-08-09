"""Example custom split script for LCO-style scaling-law experiments.

Demonstrates the ``ExternalSplitCreator`` interface used by ``MuDataSplitter``.
Define ``create_splits(mudataset, params) -> list[SplitMasks]`` at module level.

Usage::

    from drevalpy.data.splitters import get_splitter

    splitter = get_splitter(test_mode)
    folds = splitter(
        my_mudataset,
        mode="LCO",
        n_splits=1,
        external_splitter="examples/custom_split_lco_fraction.py",
    )

Custom split scripts execute as local Python code. drevalpy validates obvious
overlap/leakage for the selected ``test_mode``, but cannot guarantee that the split
answers your scientific question.
"""

from __future__ import annotations

import numpy as np

from drevalpy.data.structures import MuDataLike, SplitMasks, SplitParams

TEST_FRACTION = 0.2


def create_splits(
    mudataset: MuDataLike,
    params: SplitParams,
) -> list[SplitMasks]:
    """Return one LCO-style split with configurable train/validation/test cell-line groups.

    :param mudataset: object exposing cell_line_ids, drug_ids, response_matrix, get_tissue
    :param params: pipeline split settings (seed, validation ratio, fold count, etc.)
    :returns: list containing one SplitMasks with cell-line index arrays
    """
    rng = np.random.default_rng(params.random_state)
    unique_cell_lines = np.unique(mudataset.cell_line_ids)
    shuffled = rng.permutation(unique_cell_lines)

    n_test = max(1, int(len(shuffled) * TEST_FRACTION))
    n_val = max(1, int(len(shuffled) * params.validation_ratio))
    n_val = min(n_val, len(shuffled) - n_test - 1)

    test_cls = set(shuffled[:n_test].tolist())
    val_cls = set(shuffled[n_test : n_test + n_val].tolist())
    train_cls = set(shuffled[n_test + n_val :].tolist())

    all_cl_ids = mudataset.cell_line_ids
    all_drug_ids = mudataset.drug_ids
    n_drugs = len(all_drug_ids)

    train_cl_mask = np.isin(all_cl_ids, list(train_cls))
    val_cl_mask = np.isin(all_cl_ids, list(val_cls))
    test_cl_mask = np.isin(all_cl_ids, list(test_cls))

    train_cl_idx = np.where(train_cl_mask)[0]
    val_cl_idx = np.where(val_cl_mask)[0]
    test_cl_idx = np.where(test_cl_mask)[0]

    train_pairs = np.array([[c, d] for c in train_cl_idx for d in range(n_drugs)])
    val_pairs = np.array([[c, d] for c in val_cl_idx for d in range(n_drugs)]) if len(val_cl_idx) > 0 else np.empty((0, 2), dtype=np.intp)
    test_pairs = np.array([[c, d] for c in test_cl_idx for d in range(n_drugs)])

    return [
        SplitMasks(
            train=train_pairs,
            test=test_pairs,
            val=val_pairs,
        )
    ]
