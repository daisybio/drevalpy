"""Tests for drevalpy.datasets.splits.providers."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pytest

from drevalpy.datasets.splits import (
    MANIFEST_FILENAME,
    SplitError,
    SplitParams,
    create_and_record_splits,
    create_splits,
    load_external_splitter,
    read_split_manifest,
    run_builtin_splitter,
    run_external_splitter,
    validate_split_label,
)
from tests.datasets.split_helpers import sample_dataset


def test_validate_split_label_rejects_path_separators() -> None:
    """Reject split labels that contain path separators."""
    with pytest.raises(SplitError):
        validate_split_label("scaling/lco")


def test_load_external_splitter_requires_create_splits(tmp_path: Path) -> None:
    """
    Require a module-level create_splits function in external scripts.

    :param tmp_path: Temporary path provided by pytest.
    """
    script = tmp_path / "bad.py"
    script.write_text("def other():\n    pass\n", encoding="utf-8")
    with pytest.raises(AttributeError, match="create_splits"):
        load_external_splitter(script)


def test_run_external_splitter_adds_early_stopping_roles(tmp_path: Path) -> None:
    """
    Add early-stopping roles when running an external splitter script.

    :param tmp_path: Temporary path provided by pytest.
    """
    script = tmp_path / "splitter.py"
    script.write_text(
        """
import numpy as np
from drevalpy.datasets.dataset import DrugResponseDataset

def create_splits(response_data, params):
    mask_train = response_data.cell_line_ids == "CL-0"
    mask_val = response_data.cell_line_ids == "CL-1"
    mask_test = response_data.cell_line_ids == "CL-2"
    def pick(mask):
        return DrugResponseDataset(
            response=response_data.response[mask],
            cell_line_ids=response_data.cell_line_ids[mask],
            drug_ids=response_data.drug_ids[mask],
            tissues=response_data.tissue[mask],
            dataset_name=response_data.dataset_name,
        )
    return [{"train": pick(mask_train), "validation": pick(mask_val), "test": pick(mask_test)}]
""",
        encoding="utf-8",
    )
    dataset = sample_dataset(n_cell_lines=3, n_drugs=2)
    params = SplitParams(
        test_mode="LCO",
        n_cv_splits=1,
        validation_ratio=0.25,
        random_state=7,
        split_early_stopping=True,
    )
    splits, metadata = run_external_splitter(dataset, script, params)
    assert "validation_es" in splits[0]
    assert "early_stopping" in splits[0]
    assert metadata[0]["split_index"] == 0
    assert "test_mode" not in metadata[0]


def test_run_builtin_splitter_returns_split_level_metadata() -> None:
    """Built-in split creation returns fold dicts and per-split metadata only."""
    dataset = sample_dataset(n_cell_lines=4, n_drugs=2)
    params = SplitParams(
        test_mode="LPO",
        n_cv_splits=2,
        validation_ratio=0.2,
        random_state=11,
        split_early_stopping=False,
    )
    splits, metadata = run_builtin_splitter(dataset, params)
    assert len(splits) == 2
    assert {"train", "validation", "test"}.issubset(splits[0])
    assert metadata[0] == {"split_index": 0}
    assert metadata[1] == {"split_index": 1}


def test_create_splits_builtin_matches_run_builtin_splitter() -> None:
    """Shared split provider uses built-in splitting when no external script is given."""
    dataset = sample_dataset(n_cell_lines=4, n_drugs=2)
    splits, metadata = create_splits(
        dataset,
        test_mode="LPO",
        n_cv_splits=2,
        validation_ratio=0.2,
        random_state=11,
        split_early_stopping=False,
    )
    assert len(splits) == 2
    assert metadata[0]["split_index"] == 0


def test_create_and_record_splits_attaches_splits_and_writes_manifest(tmp_path: Path) -> None:
    """
    Create splits, attach them to the dataset, and persist the manifest.

    :param tmp_path: Temporary path provided by pytest.
    """
    dataset = sample_dataset(n_cell_lines=4, n_drugs=2)
    splits, metadata = create_and_record_splits(
        dataset,
        split_path=tmp_path,
        split_label="scaling-lco",
        test_mode="LPO",
        n_cv_splits=2,
        validation_ratio=0.2,
        random_state=11,
        split_early_stopping=False,
    )
    assert len(splits) == 2
    assert dataset.cv_splits is splits
    assert metadata[0]["split_index"] == 0
    manifest = read_split_manifest(tmp_path / MANIFEST_FILENAME)
    assert manifest is not None
    assert manifest["split_label"] == "scaling-lco"
    assert manifest["test_mode"] == "LPO"


def test_make_cv_pkls_with_builtin_splitter_writes_manifest(tmp_path: Path) -> None:
    """
    Generate split pickle files and a manifest for built-in splitting.

    :param tmp_path: Temporary path provided by pytest.
    """
    from drevalpy.cli_run_cv import run_cv_split

    dataset = sample_dataset(n_cell_lines=12, n_drugs=2)
    response_pkl = tmp_path / "response.pkl"
    with response_pkl.open("wb") as handle:
        pickle.dump(dataset, handle)

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        run_cv_split(
            response=str(response_pkl.name),
            n_cv_splits=2,
            test_mode="LCO",
            validation_ratio=0.33,
        )
        assert (tmp_path / "split_0.pkl").is_file()
        manifest = read_split_manifest(tmp_path / MANIFEST_FILENAME)
        assert manifest is not None
        assert manifest["test_mode"] == "LCO"
        assert manifest["split_label"] == "LCO"
        assert manifest["n_cv_splits"] == 2
    finally:
        os.chdir(cwd)


def test_make_cv_pkls_with_external_splitter(tmp_path: Path) -> None:
    """
    Generate split pickle files from an external splitter via run_cv_split.

    :param tmp_path: Temporary path provided by pytest.
    """
    from drevalpy.cli_run_cv import run_cv_split

    dataset = sample_dataset(n_cell_lines=4, n_drugs=2)
    response_pkl = tmp_path / "response.pkl"
    with response_pkl.open("wb") as handle:
        pickle.dump(dataset, handle)

    script = tmp_path / "splitter.py"
    script.write_text(
        """
from drevalpy.datasets.dataset import DrugResponseDataset

def create_splits(response_data, params):
    train = response_data.cell_line_ids == "CL-0"
    val = response_data.cell_line_ids == "CL-1"
    test = response_data.cell_line_ids == "CL-2"
    def subset(mask):
        return DrugResponseDataset(
            response=response_data.response[mask],
            cell_line_ids=response_data.cell_line_ids[mask],
            drug_ids=response_data.drug_ids[mask],
            tissues=response_data.tissue[mask],
            dataset_name=response_data.dataset_name,
        )
    return [
        {"train": subset(train), "validation": subset(val), "test": subset(test)},
        {
            "train": subset(response_data.cell_line_ids == "CL-3"),
            "validation": subset(val),
            "test": subset(test),
        },
    ]
""",
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        run_cv_split(
            response=str(response_pkl.name),
            n_cv_splits=2,
            test_mode="LCO",
            custom_splitter_path=str(script),
        )
        assert (tmp_path / "split_0.pkl").is_file()
        assert (tmp_path / "split_1.pkl").is_file()
        with (tmp_path / "split_0.pkl").open("rb") as handle:
            split = pickle.load(handle)
        assert {"train", "validation", "test", "validation_es", "early_stopping"}.issubset(split)
    finally:
        os.chdir(cwd)


def test_run_external_splitter_forwards_params_to_script(tmp_path: Path) -> None:
    """
    Forward SplitParams fields to create_splits scripts.

    :param tmp_path: Temporary path provided by pytest.
    """
    script = tmp_path / "params_splitter.py"
    script.write_text(
        """
from drevalpy.datasets.dataset import DrugResponseDataset

def create_splits(response_data, params):
    def pick(cell_line):
        mask = response_data.cell_line_ids == cell_line
        return DrugResponseDataset(
            response=response_data.response[mask],
            cell_line_ids=response_data.cell_line_ids[mask],
            drug_ids=response_data.drug_ids[mask],
            tissues=response_data.tissue[mask],
            dataset_name=response_data.dataset_name,
        )
    return [{
        "train": pick("CL-0"),
        "validation": pick("CL-1"),
        "test": pick("CL-2"),
        "metadata": {
            "random_state": params.random_state,
            "validation_ratio": params.validation_ratio,
            "n_cv_splits": params.n_cv_splits,
            "test_mode": params.test_mode,
            "split_early_stopping": params.split_early_stopping,
        },
    }]
""",
        encoding="utf-8",
    )
    dataset = sample_dataset(n_cell_lines=3, n_drugs=2)
    params = SplitParams(
        test_mode="LCO",
        n_cv_splits=3,
        validation_ratio=0.2,
        random_state=99,
        split_early_stopping=False,
    )
    splits, metadata = run_external_splitter(dataset, script, params)
    assert metadata[0]["random_state"] == 99
    assert metadata[0]["validation_ratio"] == 0.2
    assert metadata[0]["n_cv_splits"] == 3
    assert metadata[0]["test_mode"] == "LCO"
    assert metadata[0]["split_early_stopping"] is False
    assert "validation_es" not in splits[0]
    assert "early_stopping" not in splits[0]
