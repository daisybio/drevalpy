"""Tests for one-hot identity and tissue featurizers."""

from __future__ import annotations

import numpy as np
import pytest

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
    assert list(blocks["tissue_categories"].values) == ["lung", "skin"]


def test_tissue_strict_missing_raises() -> None:
    features = FeatureDataset(
        features={
            "cl1": {"tissue": np.array(["lung"])},
            "cl2": {"gene_expression": np.array([1.0])},
        }
    )
    featurizer = TissueFeaturizer(allow_missing=False)
    with pytest.raises(ValueError, match="requires tissue"):
        featurizer.fit(features, entity_ids=np.array(["cl1", "cl2"], dtype=str))


def test_tissue_allow_missing_partial_rows_are_zero() -> None:
    features = FeatureDataset(
        features={
            "cl1": {"tissue": np.array(["lung"])},
            "cl2": {"gene_expression": np.array([1.0])},
        }
    )
    featurizer = TissueFeaturizer(allow_missing=True)
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (2, 1)
    assert matrix[0, 0] == 1.0
    assert matrix[1, 0] == 0.0


def test_tissue_allow_missing_fully_absent_is_empty() -> None:
    features = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([1.0])},
            "cl2": {"gene_expression": np.array([2.0])},
        }
    )
    featurizer = TissueFeaturizer(allow_missing=True)
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (2, 0)
    assert featurizer.output_dim == 0
