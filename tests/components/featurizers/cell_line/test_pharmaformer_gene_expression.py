"""Tests for PharmaFormer gene preprocessing."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.cell_line.pharmaformer_gene_expression import (
    PharmaFormerGeneExpressionFeaturizer,
)
from tests.conftest import MockFeatureSource


def test_pharmaformer_gene_expression_round_trips_state() -> None:
    features = MockFeatureSource(
        {f"cl{i}": {"gene_expression": np.array([i, i + 2], dtype=np.float32)} for i in range(3)},
        meta_info={"gene_expression": ["a", "b"]},
    )
    pair_expanded_ids = np.array(["cl0", "cl1", "cl0"])
    pair_expanded_es_ids = np.array(["cl2"])
    featurizer = PharmaFormerGeneExpressionFeaturizer().fit(
        features, pair_expanded_ids=pair_expanded_ids, pair_expanded_es_ids=pair_expanded_es_ids
    )
    blocks = featurizer.transform_blocks(features, np.array(["cl0", "cl2"]))
    assert blocks["gene_expression"].feature_names == ("a", "b")
    restored = PharmaFormerGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())
    np.testing.assert_allclose(restored.transform(features, np.array(["cl0", "cl2"])), blocks["gene_expression"].values)
