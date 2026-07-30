"""Direct tests for drevalpy.data.features helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from drevalpy.data.features import iterate_features, load_and_select_gene_features, load_generic_csv
from drevalpy.datasets.dataset import FeatureDataset
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


def test_load_and_select_gene_features_preserves_gene_list_order(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "TOY"
    dataset_dir.mkdir()
    gene_dir = tmp_path / "meta" / "gene_lists"
    gene_dir.mkdir(parents=True)
    # CSV columns are B, A, C but the gene list asks for A, B, C.
    pd.DataFrame(
        {"B": [2.0], "A": [1.0], "C": [3.0]},
        index=pd.Index(["cl1"], name=CELL_LINE_IDENTIFIER),
    ).to_csv(dataset_dir / "gene_expression.csv")
    pd.DataFrame({"Symbol": ["A", "B", "C"]}).to_csv(gene_dir / "ordered_genes.csv", index=False)

    loaded = load_and_select_gene_features(
        feature_type="gene_expression",
        gene_list="ordered_genes",
        data_path=str(tmp_path),
        dataset_name="TOY",
    )
    assert list(loaded.meta_info["gene_expression"]) == ["A", "B", "C"]
    np.testing.assert_allclose(
        loaded.features["cl1"]["gene_expression"],
        np.array([1.0, 2.0, 3.0]),
    )
