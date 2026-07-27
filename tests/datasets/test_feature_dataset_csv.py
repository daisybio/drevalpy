"""Tests for feature_dataset_csv export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.feature_dataset_csv import feature_dataset_to_csv


def test_feature_dataset_to_csv_writes_view(tmp_path: Path) -> None:
    dataset = FeatureDataset(features={"id1": {"view": [1.0, 2.0]}}, meta_info={"view": ["f0", "f1"]})
    out = tmp_path / "out.csv"
    feature_dataset_to_csv(dataset, out, "sample_id", "view")
    df = pd.read_csv(out)
    assert list(df.columns) == ["sample_id", "f0", "f1"]
    assert df.iloc[0]["f1"] == 2.0
