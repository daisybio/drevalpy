"""Tests for the one-hot cell-line identity featurizer."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.cell_line.identity import CellLineIdentityFeaturizer
from tests.conftest import MockFeatureSource


def test_cell_line_identity_one_hot() -> None:
    featurizer = CellLineIdentityFeaturizer()
    features = MockFeatureSource(features={})
    entity_ids = np.array(["cl1", "cl2", "cl1"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (3, 2)
    assert matrix.dtype == np.float32
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 1.0
    assert matrix[2, 0] == 1.0


def test_cell_line_identity_emits_category_metadata_block() -> None:
    featurizer = CellLineIdentityFeaturizer()
    features = MockFeatureSource(features={})
    entity_ids = np.array(["cl2", "cl1"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)

    blocks = featurizer.transform_blocks(features, entity_ids)

    assert set(blocks) == {"identity", "identity_categories"}
    assert list(blocks["identity_categories"].values) == ["cl1", "cl2"]


def test_cell_line_identity_round_trips_state() -> None:
    features = MockFeatureSource(features={})
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = CellLineIdentityFeaturizer().fit(features, entity_ids=entity_ids)

    restored = CellLineIdentityFeaturizer()
    restored.set_state(featurizer.get_state())

    assert restored.output_dim == 2
    np.testing.assert_allclose(
        restored.transform(features, entity_ids),
        featurizer.transform(features, entity_ids),
    )
