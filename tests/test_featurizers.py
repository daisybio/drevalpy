"""Tests for drug featurizers."""

import sys
from unittest.mock import patch

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


def test_bpe_smiles_featurizer(tmp_path):
    """
    Test BPE SMILES featurizer end-to-end.

    :param tmp_path: Temporary path provided by pytest.
    """
    try:
        import drevalpy.datasets.featurizer.create_bpe_smiles_embeddings as bpe_feat
    except ImportError:
        print("subword-nmt package not installed; skipping BPE SMILES featurizer test.")
        return
    dataset = "testset"
    data_dir = tmp_path / dataset
    data_dir.mkdir(parents=True)

    # write minimal SMILES CSV with multiple SMILES for BPE learning
    df = pd.DataFrame(
        {
            "pubchem_id": ["D1", "D2", "D3", "D4", "D5"],
            "canonical_smiles": ["CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC", "C1CCC(CC1)O"],
        }
    )
    (data_dir / "drug_smiles.csv").write_text(df.to_csv(index=False))

    # run main exactly as the script would
    sys.argv = ["prog", dataset, "--data_path", str(tmp_path), "--num-symbols", "100", "--max-length", "128"]
    bpe_feat.main()

    # expected output files
    out_file = data_dir / "drug_bpe_smiles.csv"
    bpe_codes_file = data_dir / "bpe.codes"
    assert out_file.exists()
    assert bpe_codes_file.exists()

    # verify output format
    df_out = pd.read_csv(out_file)
    assert "pubchem_id" in df_out.columns
    assert df_out.pubchem_id.tolist() == ["D1", "D2", "D3", "D4", "D5"]
    # Should have 128 feature columns
    feature_cols = [col for col in df_out.columns if col.startswith("feature_")]
    assert len(feature_cols) == 128
    # Values should be numeric (character ordinals, may be stored as float in CSV)
    assert pd.api.types.is_numeric_dtype(df_out[feature_cols[0]])
