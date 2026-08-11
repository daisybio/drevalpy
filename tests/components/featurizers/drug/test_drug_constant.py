"""Tests for the drug constant featurizer."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.drug.constant import DrugConstantFeaturizer
from drevalpy.registry.drug_featurizer import get as get_drug_featurizer
from drevalpy.registry.drug_featurizer import metadata as get_drug_featurizer_metadata
from tests.conftest import MockFeatureSource


def test_drug_constant_is_ones_column() -> None:
    featurizer = DrugConstantFeaturizer()
    features = MockFeatureSource(features={"d1": {"fingerprints": np.array([1.0, 0.0])}})
    entity_ids = np.array(["d1", "d2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (2, 1)
    assert matrix.dtype == np.float32
    assert np.allclose(matrix, 1.0)
    assert featurizer.output_dim == 1


def test_drug_constant_registered() -> None:
    assert get_drug_featurizer("constant") is DrugConstantFeaturizer
    meta = get_drug_featurizer_metadata("constant")
    assert "intercept" in meta["description"].lower() or "constant" in meta["description"].lower()
