"""Tests for one-hot identity and tissue featurizers."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.cell_line.identity import CellLineIdentityFeaturizer
from drevalpy.components.featurizers.cell_line.tissue import TissueFeaturizer
from drevalpy.components.featurizers.drug.identity import DrugIdentityFeaturizer
from drevalpy.datasets.dataset import FeatureDataset


def test_cell_line_identity_one_hot() -> None:
    featurizer = CellLineIdentityFeaturizer()
    features = FeatureDataset(features={})
    entity_ids = np.array(["cl1", "cl2", "cl1"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (3, 2)
    assert matrix.dtype == np.float32
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 1.0
    assert matrix[2, 0] == 1.0


def test_drug_identity_one_hot() -> None:
    featurizer = DrugIdentityFeaturizer()
    features = FeatureDataset(features={})
    entity_ids = np.array(["d1", "d2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_tissue_one_hot() -> None:
    features = FeatureDataset(
        features={
            "cl1": {"tissue": np.array(["lung"])},
            "cl2": {"tissue": np.array(["skin"])},
        }
    )
    featurizer = TissueFeaturizer()
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (2, 2)
    blocks = featurizer.transform_blocks(features, entity_ids)
    assert "tissue_categories" in blocks
    assert list(blocks["tissue_categories"]) == ["lung", "skin"]
