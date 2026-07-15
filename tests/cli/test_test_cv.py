"""Regression tests for ``drevalpy test-cv`` in randomization mode."""

from __future__ import annotations

import os
import pathlib
import pickle

import pytest
import yaml
from typer.testing import CliRunner

from drevalpy.cli.main import app
from drevalpy.datasets.dataset import DrugResponseDataset

runner = CliRunner()


@pytest.mark.parametrize("randomization_type", ["permutation", "invariant"])
def test_randomization_cli(
    data_dir: pathlib.Path,
    sample_dataset: DrugResponseDataset,
    tmp_path: pathlib.Path,
    randomization_type: str,
) -> None:
    """Tests the functionality of the CLI call test-cv --mode randomization."""
    cv_splits = sample_dataset.split_dataset(n_cv_splits=5, mode="LCO", random_state=42)
    split = cv_splits[0]

    split_path = tmp_path / "split_0.pkl"
    with open(split_path, "wb") as fh:
        pickle.dump(split, fh)

    hpam_path = tmp_path / "best_hpam_combi_split_0.yaml"
    with open(hpam_path, "w") as fh:
        yaml.dump(
            {
                "ElasticNet_split_0": {
                    "best_hpam_combi": {
                        "cell_line_views": ["gene_expression"],
                        "drug_views": ["fingerprints"],
                        "alpha": 0.1,
                        "l1_ratio": 0.5,
                    }
                }
            },
            fh,
        )

    rand_views_path = tmp_path / "randomization_test_view_SVRC_gene_expression.yaml"
    with open(rand_views_path, "w") as fh:
        yaml.dump({"test_name": "SVRC_gene_expression", "view": "gene_expression"}, fh)

    prev_dir = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "test-cv",
                "--mode",
                "randomization",
                "--model_name",
                "ElasticNet",
                "--split_id",
                "split_0",
                "--split_dataset_path",
                str(split_path),
                "--hyperparameters_path",
                str(hpam_path),
                "--path_data",
                str(data_dir),
                "--randomization_views_path",
                str(rand_views_path),
                "--randomization_type",
                randomization_type,
                "--test_mode",
                "LCO",
            ],
        )
    finally:
        os.chdir(prev_dir)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}:\n{result.output}"

    rand_files = list(tmp_path.rglob("randomization_SVRC_gene_expression_split_0.csv"))
    assert len(rand_files) == 1, f"Expected one output CSV, found: {rand_files}"
