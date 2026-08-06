"""Tests for drevalpy.components.data_loading.feature_loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drevalpy.components.data_loading import load_cell_line_features_for_model_config
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, TISSUE_IDENTIFIER
from drevalpy.models.config import from_spec


def test_tissue_featurizer_loads_tissues(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "TOY"
    dataset_dir.mkdir()
    pd.DataFrame(
        {TISSUE_IDENTIFIER: ["lung", "breast"]},
        index=pd.Index(["cl1", "cl2"], name=CELL_LINE_IDENTIFIER),
    ).to_csv(dataset_dir / "cell_line_names.csv")

    config = from_spec("NaiveTissueMeanPredictor")
    loaded = load_cell_line_features_for_model_config(config, str(tmp_path), "TOY")
    assert set(loaded.identifiers) == {"cl1", "cl2"}
    assert loaded.features["cl1"][TISSUE_IDENTIFIER].tolist() == ["lung"]
