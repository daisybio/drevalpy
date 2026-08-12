"""Tests for the one-hot drug identity featurizer."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.drug.identity import DrugIdentityFeaturizer
from tests.conftest import MockFeatureSource


def test_drug_identity_one_hot() -> None:
    featurizer = DrugIdentityFeaturizer()
    features = MockFeatureSource(features={})
    entity_ids = np.array(["d1", "d2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_drug_identity_round_trip_state() -> None:
    featurizer = DrugIdentityFeaturizer()
    features = MockFeatureSource(features={})
    entity_ids = np.array(["d1", "d2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    restored = DrugIdentityFeaturizer()
    restored.set_state(featurizer.get_state())
    matrix = restored.transform(features, entity_ids)
    assert matrix.shape == (2, 2)
