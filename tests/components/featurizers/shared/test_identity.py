"""Tests for the shared one-hot identity featurizer (``shared/identity``)."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.shared.identity import (
    DrugIdentityFeaturizer,
    SampleIdentityFeaturizer,
    SharedIdentityFeaturizer,
)
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.registry.sample_featurizer import get as get_sample_featurizer
from tests.conftest import MockFeatureSource

_SIDE_CLASSES = [SampleIdentityFeaturizer, DrugIdentityFeaturizer]


@pytest.mark.parametrize("featurizer_cls", _SIDE_CLASSES, ids=["sample", "drug"])
def test_identity_one_hot_blocks_and_state_round_trip(featurizer_cls: type) -> None:
    features = MockFeatureSource(features={})
    entity_ids = np.array(["e2", "e1", "e2"], dtype=str)
    featurizer = featurizer_cls().fit(features, entity_ids=entity_ids)

    matrix = featurizer.transform(features, entity_ids)
    blocks = featurizer.transform_blocks(features, entity_ids)

    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix, [[0, 1], [1, 0], [0, 1]])
    assert set(blocks) == {"identity", "identity_categories"}
    assert list(blocks["identity_categories"].values) == ["e1", "e2"]
    restored = featurizer_cls()
    restored.set_state(featurizer.get_state())
    assert restored.output_dim == 2
    np.testing.assert_allclose(restored.transform(features, entity_ids), matrix)


def test_identity_registers_one_class_per_side() -> None:
    assert get_sample_featurizer("identity") is SampleIdentityFeaturizer
    assert get_drug_featurizer("identity") is DrugIdentityFeaturizer
    assert (SampleIdentityFeaturizer.side, DrugIdentityFeaturizer.side) == ("sample", "drug")
    assert all(issubclass(cls, SharedIdentityFeaturizer) for cls in _SIDE_CLASSES)
    assert SharedIdentityFeaturizer.side == ""
