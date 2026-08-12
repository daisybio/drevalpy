"""Tests for the one-hot tissue cell-line featurizer.

Mirrors :mod:`drevalpy.components.featurizers.cell_line.tissue`. These cases used
to live in ``test_identity.py`` because both featurizers share
``OneHotCategoryEncoder``; the encoder itself is covered by ``test_one_hot.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.cell_line.tissue import TissueFeaturizer
from tests.conftest import MockFeatureSource


def test_tissue_one_hot() -> None:
    features = MockFeatureSource(
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
    features = MockFeatureSource(
        features={
            "cl1": {"tissue": np.array(["lung"])},
            "cl2": {"gene_expression": np.array([1.0])},
        }
    )
    featurizer = TissueFeaturizer(allow_missing=False)
    with pytest.raises(ValueError, match="requires tissue"):
        featurizer.fit(features, entity_ids=np.array(["cl1", "cl2"], dtype=str))


def test_tissue_allow_missing_partial_rows_are_zero() -> None:
    features = MockFeatureSource(
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
    features = MockFeatureSource(
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


def test_tissue_strict_transform_of_unannotated_entity_raises() -> None:
    features = MockFeatureSource(
        features={
            "cl1": {"tissue": np.array(["lung"])},
            "cl2": {"gene_expression": np.array([1.0])},
        }
    )
    featurizer = TissueFeaturizer(allow_missing=False)
    featurizer.fit(features, entity_ids=np.array(["cl1"], dtype=str))
    with pytest.raises(ValueError, match="requires tissue"):
        featurizer.transform(features, np.array(["cl1", "cl2"], dtype=str))


def test_tissue_strict_empty_entity_set_raises() -> None:
    features = MockFeatureSource(features={})
    featurizer = TissueFeaturizer(allow_missing=False)

    with pytest.raises(ValueError, match="requires tissue"):
        featurizer.fit(features, entity_ids=np.array([], dtype=str))


def test_tissue_round_trips_state() -> None:
    features = MockFeatureSource(
        features={
            "cl1": {"tissue": np.array(["lung"])},
            "cl2": {"tissue": np.array(["skin"])},
        }
    )
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = TissueFeaturizer().fit(features, entity_ids=entity_ids)

    restored = TissueFeaturizer()
    restored.set_state(featurizer.get_state())

    np.testing.assert_allclose(
        restored.transform(features, entity_ids),
        featurizer.transform(features, entity_ids),
    )
