"""Tests for drevalpy.cli.run_cv."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from drevalpy.cli.run_cv import run_load_response
from drevalpy.data.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER
from drevalpy.utils.pickle_io import load_trusted_pickle


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

            loaded = load_trusted_pickle(work_path / "response_dataset.pkl")

            assert isinstance(loaded, pd.DataFrame)
            assert loaded.attrs.get("dataset_name") == "custom_response"
            assert list(loaded[CELL_LINE_IDENTIFIER]) == ["CL1", "CL2"]
            assert list(loaded[DRUG_IDENTIFIER]) == ["100", "200"]
