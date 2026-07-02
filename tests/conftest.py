"""Pytest configuration file for the tests directory."""

import pathlib

import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.datasets.loader import load_toyv1, load_toyv2

_TESTS_DIR = pathlib.Path(__file__).parent.resolve()
_DATA_DIR = (_TESTS_DIR.parent / "data").resolve()


@pytest.fixture(scope="session")
def data_dir() -> pathlib.Path:
    """
    Fixture to provide the path to the data directory for tests.

    :returns: path to the data directory
    """
    return _DATA_DIR


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config) -> None:
    """
    Configure pytest.

    :param config: pytest config object
    """
    # Reduce flaky plugin verbosity
    config.option.flaky_report = "none"
    config.option.tbstyle = "short"


@pytest.fixture(scope="session")
def sample_dataset(data_dir) -> DrugResponseDataset:
    """
    Sample dataset for testing individual models.

    :param data_dir: path to the data directory
    :returns: drug_response, cell_line_input, drug_input
    """
    drug_response = load_toyv1(str(data_dir))
    drug_response.remove_nan_responses()
    return drug_response


@pytest.fixture(scope="session")
def cross_study_dataset(data_dir) -> DrugResponseDataset:
    """
    Sample dataset for testing individual models.

    :param data_dir: path to the data directory
    :returns: drug_response, cell_line_input, drug_input
    """
    drug_response = load_toyv2(str(data_dir))
    drug_response.remove_nan_responses()
    return drug_response


@pytest.fixture(scope="session", autouse=True)
def ensure_bpe_features(data_dir) -> None:
    """
    Ensure BPE SMILES features are created for TOYv1 and TOYv2 before tests run.

    This fixture runs automatically before any tests to ensure that PharmaFormer
    and other models requiring BPE features have the necessary data available.

    :param data_dir: path to the data directory
    """
    path_data = str(data_dir)

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


@pytest.fixture(scope="session", autouse=True)
def ensure_precily_pathway_features(data_dir) -> None:
    """
    Ensure GSVA pathway features exist for TOYv1 and TOYv2 before tests run.

    This fixture runs automatically before any tests to ensure that Precily
    and other models requiring Precily features have the necessary data available.

    :param data_dir: path to the data directory
    """
    path_data = str(data_dir)

    try:
        from drevalpy.datasets.featurizer.create_precily_pathway_features import (
            create_precily_pathway_features,
        )
    except ImportError:
        # If gseapy is not installed, skip pathway feature creation
        # Tests that require Precily features will fail with a clear error message
        return

    # Ensure datasets are loaded first (this will download them if needed)
    try:
        load_toyv1(path_data)
        load_toyv2(path_data)
    except Exception as e:
        # If dataset loading fails, skip Precily creation
        print(f"Warning: could not load datasets for pathway feature creation: {e}")
        return

    # Create Precily features for both TOYv1 and TOYv2
    for dataset_name in ["TOYv1", "TOYv2"]:
        dataset_dir = pathlib.Path(path_data) / dataset_name
        pathway_file = dataset_dir / "pathway_features.csv"
        expr_file = dataset_dir / "gene_expression.csv"

        if pathway_file.exists():
            continue
        if not expr_file.exists():
            print(f"Warning: gene_expression.csv not found for {dataset_name}, skipping")
            continue

        # Collect gene symbols from the expression header (drop id/name columns)
        with open(expr_file, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        non_gene_cols = {"cellosaurus_id", "cell_line_name"}
        genes = [c for c in header if c not in non_gene_cols]

        # GSVA filters gene sets by min_size (default 5); build overlapping sets
        # of >=5 genes each so at least a couple survive the size filter.
        min_size = 5
        if len(genes) < min_size:
            print(f"Warning: too few genes in {dataset_name} ({len(genes)}), skipping")
            continue

        gene_sets = {
            "SYNTH_PATHWAY_A": genes[: max(min_size, len(genes) // 2)],
            "SYNTH_PATHWAY_B": genes[-max(min_size, len(genes) // 2) :],  # noqa: E203
        }

        # Write a temporary .gmt next to the dataset
        gmt_path = dataset_dir / "synthetic_pathways.gmt"
        with open(gmt_path, "w", encoding="utf-8") as f:
            for name, set_genes in gene_sets.items():
                f.write("\t".join([name, "synthetic", *set_genes]) + "\n")

        try:
            print(f"Creating synthetic GSVA pathway features for {dataset_name}...")
            create_precily_pathway_features(
                data_path=path_data,
                dataset_name=dataset_name,
                gene_sets=str(gmt_path),
                min_size=min_size,
            )
        except Exception as e:
            # Log but don't fail - let individual tests handle missing features
            print(f"Warning: could not create pathway features for {dataset_name}: {e}")
            import traceback

            traceback.print_exc()


@pytest.fixture(scope="session", autouse=True)
def ensure_precily_drug_features(data_dir) -> None:
    """
    Ensure SMILESVec drug features exist for TOYv1 and TOYv2 before tests run.

    This fixture runs automatically before any tests to ensure that Precily
    and other models requiring Precily features have the necessary data available.

    :param data_dir: path to the data directory
    """
    path_data = str(data_dir)

    # Ensure datasets are loaded first (this will download them if needed)
    try:
        load_toyv1(path_data)
        load_toyv2(path_data)
    except Exception as e:
        print(f"Warning: could not load datasets for drug feature creation: {e}")
        return

    embedding_dim = 100  # matches the SMILESVec featurizer default (dim=100)

    for dataset_name in ["TOYv1", "TOYv2"]:
        dataset_dir = pathlib.Path(path_data) / dataset_name
        smilesvec_file = dataset_dir / "drug_smilesvec.csv"
        smiles_file = dataset_dir / "drug_smiles.csv"

        if smilesvec_file.exists():
            continue
        if not smiles_file.exists():
            print(f"Warning: drug_smiles.csv not found for {dataset_name}, skipping")
            continue

        try:
            import numpy as np
            import pandas as pd

            smiles_df = pd.read_csv(smiles_file, dtype=str)
            pubchem_ids = smiles_df["pubchem_id"].astype(str).tolist()

            # Deterministic synthetic embeddings for reproducible test runs
            rng = np.random.default_rng(seed=42)
            embeddings = rng.standard_normal((len(pubchem_ids), embedding_dim)).astype(np.float32)

            out_df = pd.DataFrame(embeddings, index=pubchem_ids)
            out_df.index.name = "pubchem_id"
            print(f"Creating synthetic SMILESVec drug features for {dataset_name}...")
            out_df.to_csv(smilesvec_file)
        except Exception as e:
            print(f"Warning: could not create drug features for {dataset_name}: {e}")
