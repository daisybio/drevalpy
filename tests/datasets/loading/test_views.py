"""Tests for legacy view-string feature loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drevalpy.datasets.loading.views import load_cell_line_feature_views, load_drug_feature_views
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER, DRUG_IDENTIFIER


def test_load_cell_line_feature_views_single_custom_view(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "TOY"
    dataset_dir.mkdir()
    pd.DataFrame(
        {"feat1": [0.1, 0.2]},
        index=pd.Index(["cl1", "cl2"], name=CELL_LINE_IDENTIFIER),
    ).to_csv(dataset_dir / "custom_view.csv")

    loaded = load_cell_line_feature_views(["custom_view"], str(tmp_path), "TOY")
    assert set(loaded.identifiers) == {"cl1", "cl2"}
    assert "custom_view" in loaded.features["cl1"]


def test_load_drug_feature_views_empty_returns_none(tmp_path: Path) -> None:
    assert load_drug_feature_views([], str(tmp_path), "TOY") is None


def test_load_drug_feature_views_generic_csv(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "TOY"
    dataset_dir.mkdir()
    pd.DataFrame(
        {"feat1": [1.0, 2.0]},
        index=pd.Index(["d1", "d2"], name=DRUG_IDENTIFIER),
    ).to_csv(dataset_dir / "custom_drug.csv")

    loaded = load_drug_feature_views(["custom_drug"], str(tmp_path), "TOY")
    assert loaded is not None
    assert set(loaded.identifiers) == {"d1", "d2"}
