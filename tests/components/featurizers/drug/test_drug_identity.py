"""Tests for drug identity featurizer."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.drug.identity import DrugIdentityFeaturizer
from drevalpy.datasets.dataset import FeatureDataset


def test_drug_identity_round_trip_state() -> None:
    featurizer = DrugIdentityFeaturizer()
    features = FeatureDataset(features={})
    entity_ids = np.array(["d1", "d2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    restored = DrugIdentityFeaturizer()
    restored.set_state(featurizer.get_state())
    matrix = restored.transform(features, entity_ids)
    assert matrix.shape == (2, 2)
