"""Direct tests for drevalpy.datasets.feature_tables helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.datasets.feature_tables import iterate_features, load_generic_csv
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER


def test_iterate_features_averages_duplicate_rows() -> None:
    df = pd.DataFrame(
        {
            "gene_a": [1.0, 3.0],
            "gene_b": [2.0, 4.0],
        },
        index=["cl1", "cl1"],
    )
    features = iterate_features(df, feature_type="gene_expression")
    assert "cl1" in features
    np.testing.assert_allclose(features["cl1"]["gene_expression"], np.array([2.0, 3.0]))


def test_load_generic_csv_reads_feature_table(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "TOY"
    dataset_dir.mkdir()
    csv_path = dataset_dir / "custom_view.csv"
    pd.DataFrame(
        {"feat1": [0.1, 0.2], "feat2": [0.3, 0.4]},
        index=pd.Index(["cl1", "cl2"], name=CELL_LINE_IDENTIFIER),
    ).to_csv(csv_path)

    loaded = load_generic_csv(str(tmp_path), "TOY", "custom_view")
    assert isinstance(loaded, FeatureDataset)
    assert set(loaded.identifiers) == {"cl1", "cl2"}
    np.testing.assert_allclose(loaded.features["cl1"]["custom_view"], np.array([0.1, 0.3]))
    assert tuple(loaded.meta_info["custom_view"]) == ("feat1", "feat2")
