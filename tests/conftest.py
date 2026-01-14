"""Pytest configuration file for the tests directory."""

import os
import pathlib

import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.datasets.loader import load_toyv1, load_toyv2


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config) -> None:
    """
    Change to the tests directory and adjust pytest settings.

    :param config: pytest config object
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Reduce flaky plugin verbosity
    config.option.flaky_report = "none"
    config.option.tbstyle = "short"


@pytest.fixture(scope="session")
def sample_dataset() -> DrugResponseDataset:
    """
    Sample dataset for testing individual models.

    :returns: drug_response, cell_line_input, drug_input
    """
    path_data = str((pathlib.Path("..") / "data").resolve())
    drug_response = load_toyv1(path_data)
    drug_response.remove_nan_responses()
    return drug_response


@pytest.fixture(scope="session")
def cross_study_dataset() -> DrugResponseDataset:
    """
    Sample dataset for testing individual models.

    :returns: drug_response, cell_line_input, drug_input
    """
    path_data = str((pathlib.Path("..") / "data").resolve())
    drug_response = load_toyv2(path_data)
    drug_response.remove_nan_responses()
    return drug_response


@pytest.fixture(scope="session", autouse=True)
def ensure_bpe_features() -> None:
    """
    Ensure BPE SMILES features are created for TOYv1 and TOYv2 before tests run.

    This fixture runs automatically before any tests to ensure that PharmaFormer
    and other models requiring BPE features have the necessary data available.
    """
    # Ensure we're in the tests directory (pytest_configure should have done this)
    tests_dir = pathlib.Path(__file__).parent.resolve()
    path_data = str((tests_dir.parent / "data").resolve())

    try:
        from drevalpy.datasets.featurizer.create_pharmaformer_drug_embeddings import (
            create_pharmaformer_drug_embeddings,
        )
    except ImportError:
        # If subword-nmt is not installed, skip BPE feature creation
        # Tests that require BPE features will fail with a clear error message
        return

    # Ensure datasets are loaded first (this will download them if needed)
    try:
        load_toyv1(path_data)
        load_toyv2(path_data)
    except Exception as e:
        # If dataset loading fails, skip BPE creation
        print(f"Warning: Could not load datasets for BPE feature creation: {e}")
        return

    # Create BPE features for both TOYv1 and TOYv2
    for dataset_name in ["TOYv1", "TOYv2"]:
        dataset_dir = pathlib.Path(path_data) / dataset_name
        bpe_smiles_file = dataset_dir / "drug_bpe_smiles.csv"
        smiles_file = dataset_dir / "drug_smiles.csv"

        # Only create if it doesn't exist and if drug_smiles.csv exists
        if not bpe_smiles_file.exists():
            if not smiles_file.exists():
                print(f"Warning: drug_smiles.csv not found for {dataset_name}, skipping BPE creation")
                continue

            try:
                print(f"Creating BPE SMILES features for {dataset_name}...")
                create_pharmaformer_drug_embeddings(
                    data_path=path_data,
                    dataset_name=dataset_name,
                    num_symbols=10000,
                    max_length=128,
                )
                print(f"BPE SMILES features created for {dataset_name}")
            except Exception as e:
                # Log but don't fail - let individual tests handle missing features
                print(f"Warning: Could not create BPE features for {dataset_name}: {e}")
                import traceback

                traceback.print_exc()
