"""Tests for scaled gene-expression featurizer state."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.cell_line.scaled_gene_expression import ScaledGeneExpressionFeaturizer
from drevalpy.datasets.dataset import FeatureDataset


def test_scaled_gene_expression_output_dim_round_trips() -> None:
    features = FeatureDataset(
        features={
            "cl1": {"gene_expression": np.array([0.0, 1.0, 2.0], dtype=np.float32)},
            "cl2": {"gene_expression": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
        },
        meta_info={"gene_expression": ["g1", "g2", "g3"]},
    )
    ids = np.array(["cl1", "cl2"])
    featurizer = ScaledGeneExpressionFeaturizer()
    featurizer.fit(features, entity_ids=ids)
    assert featurizer.output_dim == 3
    matrix = featurizer.transform(features, ids)

    restored = ScaledGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())
    assert restored.output_dim == 3
    np.testing.assert_allclose(restored.transform(features, ids), matrix)
