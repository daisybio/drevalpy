"""Tests for landmark gene featurizers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drevalpy.components.featurizers.cell_line.landmark import (
    LandmarkGenesFeaturizer,
    LandmarkGenesReducedFeaturizer,
)
from drevalpy.datasets.dataset import FeatureDataset


def _features() -> FeatureDataset:
    return FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)},
            "cl2": {"gene_expression": np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)},
        },
        meta_info={"gene_expression": ["A", "B", "C", "D"]},
    )


def test_landmark_uses_symbol_column_and_persists_state(tmp_path: Path) -> None:
    gene_dir = tmp_path / "meta" / "gene_lists"
    gene_dir.mkdir(parents=True)
    pd.DataFrame({"Symbol": ["B", "D"]}).to_csv(gene_dir / "landmark_genes.csv", index=False)

    featurizer = LandmarkGenesFeaturizer(data_path=str(tmp_path), standardize=True)
    features = _features()
    ids = np.array(["cl1", "cl2"])
    featurizer.fit(features, entity_ids=ids)
    assert featurizer.output_dim == 2
    matrix = featurizer.transform(features, ids)
    assert matrix.shape == (2, 2)

    restored = LandmarkGenesFeaturizer(data_path=str(tmp_path))
    restored.set_state(featurizer.get_state())
    assert restored.output_dim == 2
    np.testing.assert_allclose(restored.transform(features, ids), matrix)


def test_landmark_reduced_uses_package_gene_list() -> None:
    featurizer = LandmarkGenesReducedFeaturizer(standardize=False)
    # Use synthetic meta that includes a couple of real reduced-list symbols.
    from drevalpy.datasets.gene_lists import gene_names_from_list_csv, resolve_gene_list_path

    symbols = gene_names_from_list_csv(resolve_gene_list_path("landmark_genes_reduced"))[:3]
    features = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.arange(len(symbols) + 1, dtype=np.float32)},
        },
        meta_info={"gene_expression": [*symbols, "NOT_A_GENE"]},
    )
    featurizer.fit(features, entity_ids=np.array(["cl1"]))
    assert featurizer.output_dim == 3


def test_landmark_fails_clearly_on_bad_column(tmp_path: Path) -> None:
    gene_dir = tmp_path / "meta" / "gene_lists"
    gene_dir.mkdir(parents=True)
    pd.DataFrame({"other": ["A"]}).to_csv(gene_dir / "landmark_genes.csv", index=False)
    featurizer = LandmarkGenesFeaturizer(data_path=str(tmp_path))
    with pytest.raises(ValueError, match="recognized gene-name column"):
        featurizer.fit(_features(), entity_ids=np.array(["cl1"]))


def test_landmark_accepts_a_path_data_path_and_serializes_it_as_a_string(tmp_path: Path) -> None:
    """A ``Path`` data_path round-trips, and is persisted as ``str`` for artifact stability.

    :param tmp_path: Temporary directory used as the dataset root.
    """
    gene_dir = tmp_path / "meta" / "gene_lists"
    gene_dir.mkdir(parents=True)
    pd.DataFrame({"Symbol": ["B", "D"]}).to_csv(gene_dir / "landmark_genes.csv", index=False)

    featurizer = LandmarkGenesFeaturizer(data_path=tmp_path, standardize=True)
    features = _features()
    ids = np.array(["cl1", "cl2"])
    featurizer.fit(features, entity_ids=ids)
    matrix = featurizer.transform(features, ids)

    state = featurizer.get_state()
    assert state["data_path"] == str(tmp_path)

    restored = LandmarkGenesFeaturizer()
    restored.set_state(state)
    assert restored._data_path == tmp_path
    np.testing.assert_allclose(restored.transform(features, ids), matrix)
