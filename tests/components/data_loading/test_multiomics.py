"""Direct tests for drevalpy.components.data_loading.multiomics helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from drevalpy.components.data_loading.multiomics import load_and_select_gene_features
from drevalpy.datasets.utils import CELL_LINE_IDENTIFIER


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
