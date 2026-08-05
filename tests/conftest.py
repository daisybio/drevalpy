"""Pytest configuration file for the tests directory."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.datasets.loader import load_dataset

_BUILTIN_MEASURE = "LN_IC50_curvecurator"

_TESTS_DIR = pathlib.Path(__file__).parent.resolve()
_DATA_DIR = (_TESTS_DIR.parent / "data").resolve()
_TOY_DATASETS = ("TOYv1", "TOYv2")


def _load_toy_datasets(path_data: str) -> bool:
    """Download TOYv1/TOYv2 once for session fixtures.

    :param path_data: path to the data directory
    :returns: False when dataset download fails
    """
    try:
        load_dataset("TOYv1", path_data, measure=_BUILTIN_MEASURE)
        load_dataset("TOYv2", path_data, measure=_BUILTIN_MEASURE)
    except Exception as exc:
        print(f"Warning: could not load TOY datasets: {exc}")
        return False
    return True


def _write_synthetic_smilesvec(
    smiles_file: pathlib.Path, output_file: pathlib.Path, *, embedding_dim: int = 100
) -> None:
    smiles_df = pd.read_csv(smiles_file, dtype=str)
    pubchem_ids = smiles_df["pubchem_id"].astype(str).tolist()
    rng = np.random.default_rng(seed=42)
    embeddings = rng.standard_normal((len(pubchem_ids), embedding_dim)).astype(np.float32)
    out_df = pd.DataFrame(embeddings, index=pubchem_ids)
    out_df.index.name = "pubchem_id"
    out_df.to_csv(output_file)


def _write_synthetic_bpe_smiles(smiles_file: pathlib.Path, output_file: pathlib.Path, *, max_length: int = 128) -> None:
    smiles_df = pd.read_csv(smiles_file, dtype=str)
    pubchem_ids = smiles_df["pubchem_id"].astype(str).tolist()
    rng = np.random.default_rng(seed=43)
    columns = [f"feature_{index}" for index in range(max_length)]
    data = rng.standard_normal((len(pubchem_ids), max_length)).astype(np.float32)
    out_df = pd.DataFrame(data, columns=columns)
    out_df.insert(0, "pubchem_id", pubchem_ids)
    out_df.to_csv(output_file, index=False)


def _ensure_bpe_smiles_features(path_data: str, dataset_name: str) -> None:
    dataset_dir = pathlib.Path(path_data) / dataset_name
    bpe_smiles_file = dataset_dir / "drug_bpe_smiles.csv"
    smiles_file = dataset_dir / "drug_smiles.csv"
    if bpe_smiles_file.exists():
        return
    if not smiles_file.exists():
        print(f"Warning: drug_smiles.csv not found for {dataset_name}, skipping BPE creation")
        return

    try:
        from drevalpy.datasets.featurizer.create_pharmaformer_drug_embeddings import (
            create_pharmaformer_drug_embeddings,
        )
    except ImportError:
        print(f"Creating synthetic BPE SMILES features for {dataset_name}...")
        _write_synthetic_bpe_smiles(smiles_file, bpe_smiles_file)
        return

    try:
        print(f"Creating BPE SMILES features for {dataset_name}...")
        create_pharmaformer_drug_embeddings(
            data_path=path_data,
            dataset_name=dataset_name,
            num_symbols=10000,
            max_length=128,
        )
        print(f"BPE SMILES features created for {dataset_name}")
    except Exception as exc:
        print(f"Warning: could not create BPE features for {dataset_name}: {exc}")
        print(f"Creating synthetic BPE SMILES features for {dataset_name}...")
        _write_synthetic_bpe_smiles(smiles_file, bpe_smiles_file)


def _ensure_smilesvec_features(path_data: str, dataset_name: str) -> None:
    dataset_dir = pathlib.Path(path_data) / dataset_name
    smilesvec_file = dataset_dir / "drug_smilesvec.csv"
    smiles_file = dataset_dir / "drug_smiles.csv"
    if smilesvec_file.exists():
        return
    if not smiles_file.exists():
        print(f"Warning: drug_smiles.csv not found for {dataset_name}, skipping")
        return

    try:
        print(f"Creating synthetic SMILESVec drug features for {dataset_name}...")
        _write_synthetic_smilesvec(smiles_file, smilesvec_file)
    except Exception as exc:
        print(f"Warning: could not create drug features for {dataset_name}: {exc}")


@pytest.fixture(scope="session")
def data_dir() -> pathlib.Path:
    """Fixture to provide the path to the data directory for tests.

    :returns: path to the data directory
    """
    return _DATA_DIR


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config) -> None:
    """Configure pytest.

    :param config: pytest config object
    """
    # Reduce flaky plugin verbosity
    config.option.flaky_report = "none"
    config.option.tbstyle = "short"


@pytest.fixture(scope="session")
def sample_dataset(data_dir) -> DrugResponseDataset:
    """Sample dataset for testing individual models.

    :param data_dir: path to the data directory
    :returns: drug_response, cell_line_input, drug_input
    """
    drug_response = load_dataset("TOYv1", path_data=str(data_dir), measure=_BUILTIN_MEASURE)
    drug_response.remove_nan_responses()
    return drug_response


@pytest.fixture(scope="session")
def cross_study_dataset(data_dir) -> DrugResponseDataset:
    """Sample dataset for testing individual models.

    :param data_dir: path to the data directory
    :returns: drug_response, cell_line_input, drug_input
    """
    drug_response = load_dataset("TOYv2", path_data=str(data_dir), measure=_BUILTIN_MEASURE)
    drug_response.remove_nan_responses()
    return drug_response


@pytest.fixture(scope="session", autouse=True)
def ensure_bpe_features(data_dir) -> None:
    """Ensure BPE SMILES features are created for TOYv1 and TOYv2 before tests run.

    This fixture runs automatically before any tests to ensure that PharmaFormer
    and other models requiring BPE features have the necessary data available.

    :param data_dir: path to the data directory
    """
    path_data = str(data_dir)
    if not _load_toy_datasets(path_data):
        return

    for dataset_name in _TOY_DATASETS:
        _ensure_bpe_smiles_features(path_data, dataset_name)


@pytest.fixture(scope="session", autouse=True)
def ensure_precily_pathway_features(data_dir) -> None:
    """Ensure GSVA pathway features exist for TOYv1 and TOYv2 before tests run.

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
    if not _load_toy_datasets(path_data):
        return

    # Create Precily features for both TOYv1 and TOYv2
    for dataset_name in _TOY_DATASETS:
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
            "SYNTH_PATHWAY_B": genes[-max(min_size, len(genes) // 2) :],
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
    """Ensure SMILESVec drug features exist for TOYv1 and TOYv2 before tests run.

    This fixture runs automatically before any tests to ensure that Precily
    and other models requiring Precily features have the necessary data available.

    :param data_dir: path to the data directory
    """
    path_data = str(data_dir)
    if not _load_toy_datasets(path_data):
        return

    for dataset_name in _TOY_DATASETS:
        _ensure_smilesvec_features(path_data, dataset_name)


@pytest.fixture(scope="session", autouse=True)
def ensure_sparsego_ontology_features(data_dir) -> None:
    """Ensure SparseGO ontology features exist for TOYv1 and TOYv2 before tests run.

    Prefers committed fixtures under ``tests/fixtures/sparsego/`` so CI does not
    depend on MyGene.info / GO network calls. Falls back to generating from
    go-basic.obo when fixtures are absent and optional deps are installed.

    :param data_dir: path to the data directory
    """
    path_data = str(data_dir)
    fixture_root = _TESTS_DIR / "fixtures" / "sparsego"

    # Ensure datasets are loaded first (this will download them if needed)
    if not _load_toy_datasets(path_data):
        return

    for dataset_name in _TOY_DATASETS:
        dataset_dir = pathlib.Path(path_data) / dataset_name
        ont_file = dataset_dir / "sparseGO_ont.txt"
        gene2ind_file = dataset_dir / "gene2ind.txt"
        if ont_file.exists() and gene2ind_file.exists():
            continue

        fixture_dir = fixture_root / dataset_name
        fixture_ont = fixture_dir / "sparseGO_ont.txt"
        fixture_gene2ind = fixture_dir / "gene2ind.txt"
        if fixture_ont.exists() and fixture_gene2ind.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)
            ont_file.write_bytes(fixture_ont.read_bytes())
            gene2ind_file.write_bytes(fixture_gene2ind.read_bytes())
            continue

        expr_file = dataset_dir / "gene_expression.csv"
        if not expr_file.exists():
            print(f"Warning: gene_expression.csv not found for {dataset_name}, skipping")
            continue

        try:
            from drevalpy.datasets.featurizer.create_sparsego_features import create_sparsego_files
        except ImportError:
            print(f"Warning: SparseGO feature generators unavailable for {dataset_name}")
            continue

        try:
            print(f"Generating SparseGO ontology features for {dataset_name} (network calls to GO/MyGene.info)...")
            create_sparsego_files(
                data_path=path_data,
                dataset_name=dataset_name,
            )
            print(f"SparseGO ontology features created for {dataset_name}")
        except Exception as e:
            # Log but don't fail - let individual tests handle missing features
            print(f"Warning: could not create SparseGO ontology features for {dataset_name}: {e}")
            import traceback

            traceback.print_exc()


@pytest.fixture(scope="session", autouse=True)
def ensure_model_drug_embeddings(
    data_dir,
    sample_dataset,
    cross_study_dataset,
    ensure_sparsego_ontology_features,
) -> None:
    """Re-ensure PharmaFormer/Precily drug embeddings after TOY datasets are loaded.

    Earlier autouse fixtures may run before the first successful TOY download on CI;
    this pass creates any still-missing embedding files once ``sample_dataset`` is ready.

    :param data_dir: path to the data directory
    :param sample_dataset: ensures TOYv1 is present
    :param cross_study_dataset: ensures TOYv2 is present
    :param ensure_sparsego_ontology_features: run after other feature fixtures
    """
    _ = sample_dataset, cross_study_dataset, ensure_sparsego_ontology_features
    path_data = str(data_dir)
    for dataset_name in _TOY_DATASETS:
        _ensure_bpe_smiles_features(path_data, dataset_name)
        _ensure_smilesvec_features(path_data, dataset_name)
