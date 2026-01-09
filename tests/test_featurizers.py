"""Tests for drug featurizers."""

import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


def test_chemberta_featurizer(tmp_path):
    """
    Test ChemBERTa featurizer end-to-end.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        import drevalpy.datasets.featurizer.create_chemberta_drug_embeddings as chemberta
    except ImportError:
        print("transformers package not installed; skipping ChemBERTa featurizer test.")
        return
    dataset = "testset"
    data_dir = tmp_path / dataset
    data_dir.mkdir(parents=True)

    # fake input CSV
    df = pd.DataFrame({"pubchem_id": ["X1"], "canonical_smiles": ["CCO"]})
    (data_dir / "drug_smiles.csv").write_text(df.to_csv(index=False))

    fake_embedding = [1.0, 2.0, 3.0]

    with patch.object(chemberta, "_smiles_to_chemberta", return_value=fake_embedding), patch.object(
        sys, "argv", ["prog", dataset, "--data_path", str(tmp_path)]
    ):

        chemberta.main()

    out_file = data_dir / "drug_chemberta_embeddings.csv"
    assert out_file.exists()

    df_out = pd.read_csv(out_file)
    assert df_out.pubchem_id.tolist() == ["X1"]
    assert df_out.iloc[0, 1:].tolist() == fake_embedding


def test_graph_featurizer(tmp_path):
    """
    Test graph featurizer end-to-end.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        import drevalpy.datasets.featurizer.create_drug_graphs as graphs
    except ImportError:
        print("rdkit package not installed; skipping graph featurizer test.")
        return
    dataset = "testset"
    data_dir = tmp_path / dataset
    data_dir.mkdir(parents=True)

    # write minimal SMILES CSV
    df = pd.DataFrame({"pubchem_id": ["D1"], "canonical_smiles": ["CCO"]})
    (data_dir / "drug_smiles.csv").write_text(df.to_csv(index=False))

    # run main exactly as the script would
    sys.argv = ["prog", dataset, "--data_path", str(tmp_path)]
    graphs.main()

    # expected output file
    out_file = data_dir / "drug_graphs" / "D1.pt"
    assert out_file.exists()


def test_molgnet_featurizer(tmp_path):
    """
    Test MolGNet featurizer end-to-end.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        import drevalpy.datasets.featurizer.create_molgnet_embeddings as molg
    except ImportError:
        print("rdkit package not installed; skipping molgnet featurizer test.")
        return
    ds = "testset"
    ds_dir = tmp_path / ds
    ds_dir.mkdir(parents=True)

    # minimal SMILES CSV
    df = pd.DataFrame({"pubchem_id": ["D1"], "canonical_smiles": ["CCO"]})
    (ds_dir / "drug_smiles.csv").write_text(df.to_csv(index=False))

    with (
        # we dont need real model weights for this test, takes too long to load
        patch("drevalpy.datasets.featurizer.create_molgnet_embeddings.torch.load", return_value={}),
        # prevent load_state_dict from complaining
        patch.object(molg.MolGNet, "load_state_dict", return_value=None),
        # cheap forward pass
        patch.object(molg.MolGNet, "forward", return_value=torch.zeros((1, 768))),
        # avoid writing pickles
        patch.object(molg.pickle, "dump", return_value=None),
        # simulate CLI
        patch.object(
            sys,
            "argv",
            ["prog", ds, "--data_path", str(tmp_path), "--checkpoint", "MolGNet.pt"],
        ),
    ):
        args = molg.parse_args()
        molg.run(args)

    # verify outputs
    assert (ds_dir / "DIPK_features/Drugs" / "MolGNet_D1.csv").exists()


def test_transcriptome_pca_featurizer(tmp_path):
    """
    Test transcriptome PCA featurizer end-to-end.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        import drevalpy.datasets.featurizer.create_transcriptome_pca as pca_feat
    except ImportError:
        print("sklearn package not installed; skipping transcriptome PCA featurizer test.")
        return

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
    with patch.object(
        sys,
        "argv",
        ["prog", dataset, "--data_path", str(tmp_path), "--n_components", "10"],
    ):
        pca_feat.main()

    # Check output files
    output_file = data_dir / "cell_line_gene_expression_pca_10.csv"
    pca_file = data_dir / "cell_line_gene_expression_pca_10_pca.pkl"
    scaler_file = data_dir / "cell_line_gene_expression_pca_10_scaler.pkl"

    assert output_file.exists()
    assert pca_file.exists()
    assert scaler_file.exists()

    # Verify output CSV structure
    df_out = pd.read_csv(output_file)
    assert "cell_line_name" in df_out.columns
    assert len(df_out.columns) == 11  # cell_line_name + 10 PC columns
    assert len(df_out) == n_cell_lines
    assert all(f"PC{i + 1}" in df_out.columns for i in range(10))
