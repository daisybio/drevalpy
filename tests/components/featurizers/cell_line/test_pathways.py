"""Tests for the Precily pathway featurizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drevalpy.components.featurizers.cell_line.pathways import PathwaysCellLineFeaturizer
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER


def test_pathway_loader_renames_generated_feature_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DREVALPY_CACHE_DIR", str(tmp_path))
    dataset_dir = tmp_path / "TOY"
    dataset_dir.mkdir()
    pd.DataFrame(
        {"pathway_a": [0.1], "pathway_b": [0.2]},
        index=pd.Index(["cl1"], name=CELL_LINE_IDENTIFIER),
    ).to_csv(dataset_dir / "pathway_features.csv")

    loaded = PathwaysCellLineFeaturizer.load_features("TOY")

    assert np.array_equal(loaded.features["cl1"]["pathways"], np.array([0.1, 0.2]))
    assert tuple(loaded.meta_info["pathways"]) == ("pathway_a", "pathway_b")
