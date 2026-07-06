"""Tests for normalized proteomics cell-line featurizer."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.cell_line.normalized_proteomics import (
    NormalizedProteomicsCellLineFeaturizer,
)
from drevalpy.datasets.dataset import FeatureDataset


def _make_features() -> FeatureDataset:
    return FeatureDataset(
        features={
            "cl1": {"proteomics": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            "cl2": {"proteomics": np.array([4.0, 5.0, 6.0], dtype=np.float32)},
            "cl3": {"proteomics": np.array([7.0, 8.0, 9.0], dtype=np.float32)},
        }
    )


def test_normalized_proteomics_fit_transform() -> None:
    featurizer = NormalizedProteomicsCellLineFeaturizer()
    features = _make_features()
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32
