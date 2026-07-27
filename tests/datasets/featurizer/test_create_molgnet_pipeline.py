"""Tests for create_molgnet_pipeline helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from drevalpy.datasets.featurizer.create_molgnet_pipeline import load_smiles_map, resolve_molgnet_dataset_dir


def test_resolve_molgnet_dataset_dir_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_molgnet_dataset_dir(str(tmp_path), "missing")


def test_load_smiles_map_reads_csv(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir()
    df = pd.DataFrame({"pubchem_id": ["D1"], "canonical_smiles": ["CCO"]})
    (ds_dir / "drug_smiles.csv").write_text(df.to_csv(index=False))
    mapping = load_smiles_map(ds_dir, "canonical_smiles", "pubchem_id")
    assert mapping == {"D1": "CCO"}
