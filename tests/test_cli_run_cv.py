"""Tests for drevalpy.cli_run_cv."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import pandas as pd

from drevalpy.cli_run_cv import run_load_response
from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER


def test_run_load_response_uses_provided_path() -> None:
    """Regression: ``run_load_response`` must read the given CSV path, not ``<stem>.csv`` in CWD."""
    with tempfile.TemporaryDirectory() as data_dir:
        csv_path = Path(data_dir) / "custom_response.csv"
        pd.DataFrame(
            {
                CELL_LINE_IDENTIFIER: ["CL1", "CL2"],
                DRUG_IDENTIFIER: ["100", "200"],
                "response": [0.1, 0.2],
            }
        ).to_csv(csv_path, index=False)

        with tempfile.TemporaryDirectory() as work_dir:
            work_path = Path(work_dir)
            before = set(work_path.iterdir())
            previous = os.getcwd()
            try:
                os.chdir(work_path)
                run_load_response(response_dataset=str(csv_path), measure="response")
            finally:
                os.chdir(previous)
            after = set(work_path.iterdir())
            assert after - before == {work_path / "response_dataset.pkl"}

            with open(work_path / "response_dataset.pkl", "rb") as handle:
                loaded = pickle.load(handle)

            assert isinstance(loaded, DrugResponseDataset)
            assert loaded.dataset_name == "custom_response"
            assert list(loaded.cell_line_ids) == ["CL1", "CL2"]
            assert list(loaded.drug_ids) == ["100", "200"]
