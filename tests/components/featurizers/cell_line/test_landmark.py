"""Tests for landmark gene featurizers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from upath import UPath

from drevalpy.components.featurizers.cell_line import gene_lists
from drevalpy.components.featurizers.cell_line.gene_lists import gene_names_from_list_csv, resolve_gene_list_path
from drevalpy.components.featurizers.cell_line.landmark import (
    LandmarkGenesFeaturizer,
    LandmarkGenesReducedFeaturizer,
)
from tests.conftest import MockFeatureSource


def _features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)},
            "cl2": {"gene_expression": np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)},
        },
        meta_info={"gene_expression": ["A", "B", "C", "D"]},
    )


def test_landmark_uses_symbol_column_and_persists_state() -> None:
    symbols = gene_names_from_list_csv(resolve_gene_list_path("landmark_genes"))[:2]
    features = MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            "cl2": {"gene_expression": np.array([3.0, 2.0, 1.0], dtype=np.float32)},
        },
        meta_info={"gene_expression": [*symbols, "NOT_A_GENE"]},
    )
    featurizer = LandmarkGenesFeaturizer(standardize=True)
    ids = np.array(["cl1", "cl2"])
    featurizer.fit(features, entity_ids=ids)
    assert featurizer.output_dim == 2
    matrix = featurizer.transform(features, ids)
    assert matrix.shape == (2, 2)

    restored = LandmarkGenesFeaturizer()
    restored.set_state(featurizer.get_state())
    assert restored.output_dim == 2
    np.testing.assert_allclose(restored.transform(features, ids), matrix)


def test_landmark_reduced_uses_package_gene_list() -> None:
    featurizer = LandmarkGenesReducedFeaturizer(standardize=False)
    symbols = gene_names_from_list_csv(resolve_gene_list_path("landmark_genes_reduced"))[:3]
    features = MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.arange(len(symbols) + 1, dtype=np.float32)},
        },
        meta_info={"gene_expression": [*symbols, "NOT_A_GENE"]},
    )
    featurizer.fit(features, entity_ids=np.array(["cl1"]))
    assert featurizer.output_dim == 3


def test_landmark_fails_clearly_on_bad_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd.DataFrame({"other": ["A"]}).to_csv(tmp_path / "landmark_genes.csv", index=False)
    monkeypatch.setattr(gene_lists, "GENE_LISTS_DIR", UPath(tmp_path))
    featurizer = LandmarkGenesFeaturizer()
    with pytest.raises(ValueError, match="recognized gene-name column"):
        featurizer.fit(_features(), entity_ids=np.array(["cl1"]))
