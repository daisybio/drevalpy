"""Tests for the Precily SMILESVec featurizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from drevalpy.components.featurizers.drug.smilesvec import SmilesVecDrugFeaturizer


def test_smilesvec_loader_renames_generated_feature_view(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "TOY"
    dataset_dir.mkdir()
    pd.DataFrame(
        {"feature_a": [0.1], "feature_b": [0.2]},
        index=pd.Index(["123"], name="pubchem_id"),
    ).to_csv(dataset_dir / "drug_smilesvec.csv")

    loaded = SmilesVecDrugFeaturizer.load_features(str(tmp_path), "TOY")

    assert np.array_equal(loaded.features["123"]["smilesvec"], np.array([0.1, 0.2]))
    assert tuple(loaded.meta_info["smilesvec"]) == ("feature_a", "feature_b")
