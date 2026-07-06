"""Tests for dense concat featurizers."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.cell_line.concat import ConcatFeaturizersCellLineFeaturizer
from drevalpy.datasets.dataset import FeatureDataset


def _make_features() -> FeatureDataset:
    return FeatureDataset(
        features={
            "cl1": {
                "gene_expression": np.array([1.0, 2.0], dtype=np.float32),
                "proteomics": np.array([10.0, 20.0, 30.0], dtype=np.float32),
            },
            "cl2": {
                "gene_expression": np.array([3.0, 4.0], dtype=np.float32),
                "proteomics": np.array([40.0, 50.0, 60.0], dtype=np.float32),
            },
        }
    )


def test_concat_uses_distinct_block_labels_for_same_name_different_views() -> None:
    featurizer = ConcatFeaturizersCellLineFeaturizer(
        featurizers=[
            {"name": "pca", "view": "gene_expression", "hyperparameters": {"n_components": 1}},
            {"name": "pca", "view": "proteomics", "hyperparameters": {"n_components": 1}},
        ],
        registry="cell_line",
    )
    features = _make_features()
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    blocks = featurizer.transform_blocks(features, entity_ids)
    assert set(blocks) == {"pca[expression]", "pca[proteomics]"}
    assert featurizer.block_dims == {"pca[expression]": 1, "pca[proteomics]": 1}
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (2, 2)
