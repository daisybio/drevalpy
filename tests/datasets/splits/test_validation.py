"""Tests for drevalpy.datasets.splits.validation."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.datasets.splits import SplitError, validate_splits
from tests.datasets.split_helpers import role_from_groups, sample_dataset


def test_validate_lco_rejects_shared_cell_line() -> None:
    """Reject LCO splits that share cell lines across roles."""
    dataset = sample_dataset()
    split = role_from_groups(
        dataset,
        train_groups={"CL-0", "CL-1"},
        val_groups={"CL-1", "CL-2"},
        test_groups={"CL-3"},
        group_col="cell_line",
    )
    with pytest.raises(SplitError, match="overlap|leakage"):
        validate_splits([split], "LCO")


def test_validate_ldo_rejects_shared_drug() -> None:
    """Reject LDO splits that share drugs across roles."""
    dataset = sample_dataset()
    split = role_from_groups(
        dataset,
        train_groups={"D-0", "D-1"},
        val_groups={"D-1", "D-2"},
        test_groups={"D-3"},
        group_col="drug",
    )
    with pytest.raises(SplitError, match="overlap|leakage"):
        validate_splits([split], "LDO")


def test_validate_lto_requires_tissue_and_disjointness() -> None:
    """Accept valid LTO splits with disjoint tissue groups."""
    dataset = sample_dataset()
    split = role_from_groups(
        dataset,
        train_groups={"T-0"},
        val_groups={"T-1"},
        test_groups={"T-2"},
        group_col="tissue",
    )
    validated, metadata = validate_splits([split], "LTO")
    assert len(validated) == 1
    assert metadata[0]["split_index"] == 0


def test_validate_lpo_rejects_shared_pair() -> None:
    """Reject LPO splits that share cell-line/drug pairs across roles."""
    split = {
        "train": DrugResponseDataset(
            response=np.array([1.0]),
            cell_line_ids=np.array(["CL-0"]),
            drug_ids=np.array(["D-0"]),
            dataset_name="testset",
        ),
        "validation": DrugResponseDataset(
            response=np.array([2.0]),
            cell_line_ids=np.array(["CL-0"]),
            drug_ids=np.array(["D-0"]),
            dataset_name="testset",
        ),
        "test": DrugResponseDataset(
            response=np.array([3.0]),
            cell_line_ids=np.array(["CL-1"]),
            drug_ids=np.array(["D-1"]),
            dataset_name="testset",
        ),
    }
    with pytest.raises(SplitError, match="LPO leakage"):
        validate_splits([split], "LPO")
