"""Tests for the shared constant (intercept) featurizer mixin (``_constant``)."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers._constant import ConstantFeaturizerMixin
from drevalpy.components.featurizers.shared.constant import SampleConstantFeaturizer
from tests.conftest import MockFeatureSource


def test_constant_needs_no_source_views_and_is_a_ones_column() -> None:
    featurizer = SampleConstantFeaturizer()
    features = MockFeatureSource(features={})
    ids = np.array(["cl1", "cl2", "cl3"], dtype=str)
    assert issubclass(SampleConstantFeaturizer, ConstantFeaturizerMixin)
    assert featurizer.output_dim == 1
    featurizer.fit(features, entity_ids=ids)

    blocks = featurizer.transform_blocks(features, ids)

    assert list(blocks) == ["constant"]
    np.testing.assert_allclose(blocks["constant"].values, np.ones((3, 1), dtype=np.float32))


def test_constant_state_round_trip_is_a_no_op() -> None:
    featurizer = SampleConstantFeaturizer()
    features = MockFeatureSource(features={})
    ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids=ids)
    restored = SampleConstantFeaturizer()
    restored.set_state(featurizer.get_state())

    assert featurizer.get_state() == {}
    np.testing.assert_allclose(restored.transform(features, ids), featurizer.transform(features, ids))
