"""Tests for drug and cell line featurizers."""

import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch


def test_chemberta_featurizer(tmp_path):
    """
    Test ChemBERTa featurizer end-to-end.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        from drevalpy.datasets.featurizer import ChemBERTaFeaturizer
    except ImportError:
        pytest.skip("transformers package not installed; skipping ChemBERTa featurizer test.")

    dataset = "testset"
    data_dir = tmp_path / dataset
    data_dir.mkdir(parents=True)

    # fake input CSV
    df = pd.DataFrame({"pubchem_id": ["X1"], "canonical_smiles": ["CCO"]})
    (data_dir / "drug_smiles.csv").write_text(df.to_csv(index=False))

    fake_embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    featurizer = ChemBERTaFeaturizer(device="cpu")

    with patch.object(featurizer, "featurize", return_value=fake_embedding):
        result = featurizer.generate_embeddings(str(tmp_path), dataset)

    out_file = data_dir / "drug_chemberta_embeddings.csv"
    assert out_file.exists()

    df_out = pd.read_csv(out_file)
    assert df_out.pubchem_id.tolist() == ["X1"]
    assert df_out.iloc[0, 1:].tolist() == fake_embedding.tolist()

    # Test that FeatureDataset is returned correctly
    assert "X1" in result.features
    assert "chemberta_embeddings" in result.features["X1"]


def test_chemberta_featurizer_cli(tmp_path):
    """
    Test ChemBERTa featurizer CLI entry point.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        from drevalpy.datasets.featurizer.drug import chemberta
    except ImportError:
        pytest.skip("transformers package not installed; skipping ChemBERTa featurizer CLI test.")

    dataset = "testset"
    data_dir = tmp_path / dataset
    data_dir.mkdir(parents=True)

    # fake input CSV
    df = pd.DataFrame({"pubchem_id": ["X1"], "canonical_smiles": ["CCO"]})
    (data_dir / "drug_smiles.csv").write_text(df.to_csv(index=False))

    fake_embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with (
        patch.object(chemberta.ChemBERTaFeaturizer, "featurize", return_value=fake_embedding),
        patch.object(sys, "argv", ["prog", dataset, "--data_path", str(tmp_path)]),
    ):
        chemberta.main()

    out_file = data_dir / "drug_chemberta_embeddings.csv"
    assert out_file.exists()


def test_graph_featurizer(tmp_path):
    """
    Test graph featurizer end-to-end.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        from drevalpy.datasets.featurizer import DrugGraphFeaturizer
    except ImportError:
        pytest.skip("rdkit package not installed; skipping graph featurizer test.")

    dataset = "testset"
    data_dir = tmp_path / dataset
    data_dir.mkdir(parents=True)

    # write minimal SMILES CSV
    df = pd.DataFrame({"pubchem_id": ["D1"], "canonical_smiles": ["CCO"]})
    (data_dir / "drug_smiles.csv").write_text(df.to_csv(index=False))

    featurizer = DrugGraphFeaturizer()
    result = featurizer.generate_embeddings(str(tmp_path), dataset)

    # expected output file
    out_file = data_dir / "drug_graphs" / "D1.pt"
    assert out_file.exists()

    # Test that FeatureDataset is returned correctly
    assert "D1" in result.features
    assert "drug_graphs" in result.features["D1"]


def test_graph_featurizer_cli(tmp_path):
    """
    Test graph featurizer CLI entry point.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        from drevalpy.datasets.featurizer.drug import drug_graph
    except ImportError:
        pytest.skip("rdkit package not installed; skipping graph featurizer CLI test.")

    dataset = "testset"
    data_dir = tmp_path / dataset
    data_dir.mkdir(parents=True)

    # write minimal SMILES CSV
    df = pd.DataFrame({"pubchem_id": ["D1"], "canonical_smiles": ["CCO"]})
    (data_dir / "drug_smiles.csv").write_text(df.to_csv(index=False))

    with patch.object(sys, "argv", ["prog", dataset, "--data_path", str(tmp_path)]):
        drug_graph.main()

    # expected output file
    out_file = data_dir / "drug_graphs" / "D1.pt"
    assert out_file.exists()


def test_molgnet_featurizer(tmp_path):
    """
    Test MolGNet featurizer end-to-end.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        from drevalpy.datasets.featurizer import MolGNetFeaturizer
        from drevalpy.datasets.featurizer.drug import molgnet
    except ImportError:
        pytest.skip("rdkit package not installed; skipping molgnet featurizer test.")

    ds = "testset"
    ds_dir = tmp_path / ds
    ds_dir.mkdir(parents=True)

    # minimal SMILES CSV
    df = pd.DataFrame({"pubchem_id": ["D1"], "canonical_smiles": ["CCO"]})
    (ds_dir / "drug_smiles.csv").write_text(df.to_csv(index=False))

    # Create a fake checkpoint file
    checkpoint_path = str(tmp_path / "MolGNet.pt")

    featurizer = MolGNetFeaturizer(checkpoint_path=checkpoint_path, device="cpu")

    with (
        # we dont need real model weights for this test, takes too long to load
        patch("drevalpy.datasets.featurizer.drug.molgnet.torch.load", return_value={}),
        # prevent load_state_dict from complaining
        patch.object(molgnet.MolGNet, "load_state_dict", return_value=None),
        # cheap forward pass
        patch.object(molgnet.MolGNet, "forward", return_value=torch.zeros((1, 768))),
    ):
        result = featurizer.generate_embeddings(str(tmp_path), ds)

    # verify outputs
    assert (ds_dir / "DIPK_features/Drugs" / "MolGNet_D1.csv").exists()
    assert (ds_dir / "MolGNet_dict.pkl").exists()

    # Test that FeatureDataset is returned correctly
    assert "D1" in result.features
    assert "molgnet_embeddings" in result.features["D1"]


def test_molgnet_featurizer_cli(tmp_path):
    """
    Test MolGNet featurizer CLI entry point.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        from drevalpy.datasets.featurizer.drug import molgnet
    except ImportError:
        pytest.skip("rdkit package not installed; skipping molgnet featurizer CLI test.")

    ds = "testset"
    ds_dir = tmp_path / ds
    ds_dir.mkdir(parents=True)

    # minimal SMILES CSV
    df = pd.DataFrame({"pubchem_id": ["D1"], "canonical_smiles": ["CCO"]})
    (ds_dir / "drug_smiles.csv").write_text(df.to_csv(index=False))

    with (
        # we dont need real model weights for this test, takes too long to load
        patch("drevalpy.datasets.featurizer.drug.molgnet.torch.load", return_value={}),
        # prevent load_state_dict from complaining
        patch.object(molgnet.MolGNet, "load_state_dict", return_value=None),
        # cheap forward pass
        patch.object(molgnet.MolGNet, "forward", return_value=torch.zeros((1, 768))),
        # simulate CLI
        patch.object(
            sys,
            "argv",
            ["prog", ds, "--data_path", str(tmp_path), "--checkpoint", str(tmp_path / "MolGNet.pt")],
        ),
    ):
        molgnet.main()

    # verify outputs
    assert (ds_dir / "DIPK_features/Drugs" / "MolGNet_D1.csv").exists()


def test_pca_featurizer(tmp_path):
    """
    Test transcriptome PCA featurizer end-to-end.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        from drevalpy.datasets.featurizer import PCAFeaturizer
    except ImportError:
        pytest.skip("sklearn package not installed; skipping transcriptome PCA featurizer test.")

    dataset = "testset"
    data_dir = tmp_path / dataset
    data_dir.mkdir(parents=True)

    # Create fake gene expression CSV
    # Format: rows are cell lines, columns are genes
    n_cell_lines = 10
    n_genes = 100
    cell_line_names = [f"CL{i}" for i in range(n_cell_lines)]
    gene_names = [f"GENE{i}" for i in range(n_genes)]

    # Generate some random gene expression data
    np.random.seed(42)
    ge_data = np.random.randn(n_cell_lines, n_genes).astype(np.float32)

    ge_df = pd.DataFrame(ge_data, index=cell_line_names, columns=gene_names)
    ge_df.index.name = "cell_line_name"
    ge_df = ge_df.reset_index()

    (data_dir / "gene_expression.csv").write_text(ge_df.to_csv(index=False))

    # Run the featurizer
    featurizer = PCAFeaturizer(n_components=10)
    result = featurizer.generate_embeddings(str(tmp_path), dataset)

    # Check output files
    output_file = data_dir / "cell_line_gene_expression_pca_10.csv"
    model_file = data_dir / "cell_line_gene_expression_pca_10_models.pkl"

    assert output_file.exists()
    assert model_file.exists()

    # Verify output CSV structure
    df_out = pd.read_csv(output_file)
    assert "cell_line_name" in df_out.columns
    assert len(df_out.columns) == 11  # cell_line_name + 10 PC columns
    assert len(df_out) == n_cell_lines

    # Test that FeatureDataset is returned correctly
    assert "CL0" in result.features
    assert "gene_expression_pca" in result.features["CL0"]
    assert result.features["CL0"]["gene_expression_pca"].shape == (10,)


def test_pca_featurizer_cli(tmp_path):
    """
    Test transcriptome PCA featurizer CLI entry point.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        from drevalpy.datasets.featurizer.cell_line import pca
    except ImportError:
        pytest.skip("sklearn package not installed; skipping transcriptome PCA featurizer CLI test.")

    dataset = "testset"
    data_dir = tmp_path / dataset
    data_dir.mkdir(parents=True)

    # Create fake gene expression CSV
    n_cell_lines = 10
    n_genes = 100
    cell_line_names = [f"CL{i}" for i in range(n_cell_lines)]
    gene_names = [f"GENE{i}" for i in range(n_genes)]

    np.random.seed(42)
    ge_data = np.random.randn(n_cell_lines, n_genes).astype(np.float32)

    ge_df = pd.DataFrame(ge_data, index=cell_line_names, columns=gene_names)
    ge_df.index.name = "cell_line_name"
    ge_df = ge_df.reset_index()

    (data_dir / "gene_expression.csv").write_text(ge_df.to_csv(index=False))

    # Run the featurizer via CLI
    with patch.object(
        sys,
        "argv",
        ["prog", dataset, "--data_path", str(tmp_path), "--n_components", "10"],
    ):
        pca.main()

    # Check output files
    output_file = data_dir / "cell_line_gene_expression_pca_10.csv"
    assert output_file.exists()


def test_pca_featurizer_load_or_generate(tmp_path):
    """
    Test that load_or_generate loads existing embeddings or generates new ones.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        from drevalpy.datasets.featurizer import PCAFeaturizer
    except ImportError:
        pytest.skip("sklearn package not installed; skipping PCA featurizer test.")

    dataset = "testset"
    data_dir = tmp_path / dataset
    data_dir.mkdir(parents=True)

    # Create fake gene expression CSV
    n_cell_lines = 10
    n_genes = 100
    cell_line_names = [f"CL{i}" for i in range(n_cell_lines)]
    gene_names = [f"GENE{i}" for i in range(n_genes)]

    np.random.seed(42)
    ge_data = np.random.randn(n_cell_lines, n_genes).astype(np.float32)

    ge_df = pd.DataFrame(ge_data, index=cell_line_names, columns=gene_names)
    ge_df.index.name = "cell_line_name"
    ge_df = ge_df.reset_index()

    (data_dir / "gene_expression.csv").write_text(ge_df.to_csv(index=False))

    # First call should generate embeddings
    featurizer1 = PCAFeaturizer(n_components=10)
    result1 = featurizer1.load_or_generate(str(tmp_path), dataset)

    # Second call should load existing embeddings
    featurizer2 = PCAFeaturizer(n_components=10)
    result2 = featurizer2.load_or_generate(str(tmp_path), dataset)

    # Results should be the same
    assert set(result1.features.keys()) == set(result2.features.keys())
    for cell_line_id in result1.features:
        np.testing.assert_array_almost_equal(
            result1.features[cell_line_id]["gene_expression_pca"],
            result2.features[cell_line_id]["gene_expression_pca"],
        )


def test_chemberta_featurizer_load_or_generate(tmp_path):
    """
    Test that load_or_generate loads existing embeddings or generates new ones.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        from drevalpy.datasets.featurizer import ChemBERTaFeaturizer
    except ImportError:
        pytest.skip("transformers package not installed; skipping ChemBERTa featurizer test.")

    dataset = "testset"
    data_dir = tmp_path / dataset
    data_dir.mkdir(parents=True)

    # fake input CSV
    df = pd.DataFrame({"pubchem_id": ["X1", "X2"], "canonical_smiles": ["CCO", "CC"]})
    (data_dir / "drug_smiles.csv").write_text(df.to_csv(index=False))

    fake_embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    featurizer1 = ChemBERTaFeaturizer(device="cpu")

    # First call should generate embeddings
    with patch.object(featurizer1, "featurize", return_value=fake_embedding):
        result1 = featurizer1.load_or_generate(str(tmp_path), dataset)

    # Second call should load existing embeddings (no mock needed)
    featurizer2 = ChemBERTaFeaturizer(device="cpu")
    result2 = featurizer2.load_or_generate(str(tmp_path), dataset)

    # Results should be the same
    assert set(result1.features.keys()) == set(result2.features.keys())
    for drug_id in result1.features:
        np.testing.assert_array_almost_equal(
            result1.features[drug_id]["chemberta_embeddings"],
            result2.features[drug_id]["chemberta_embeddings"],
        )
