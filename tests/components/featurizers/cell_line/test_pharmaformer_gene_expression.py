"""Tests for PharmaFormer gene preprocessing."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers.cell_line.pharmaformer_gene_expression import (
    PharmaFormerGeneExpressionFeaturizer,
)
from tests.conftest import MockFeatureSource


def test_pharmaformer_gene_expression_round_trips_state() -> None:
    features = MockFeatureSource(
        {f"cl{i}": {"gene_expression": np.array([i, i + 2], dtype=np.float32)} for i in range(3)},
        meta_info={"gene_expression": ["a", "b"]},
    )
    context = FeaturizerFitContext(
        unique_train_ids=np.array(["cl0", "cl1"]),
        pair_expanded_train_ids=np.array(["cl0", "cl1", "cl0"]),
        unique_early_stopping_ids=np.array(["cl2"]),
        pair_expanded_early_stopping_ids=np.array(["cl2"]),
        side="cell_line",
    )
    featurizer = PharmaFormerGeneExpressionFeaturizer().fit(features, context=context)
    blocks = featurizer.transform_blocks(features, np.array(["cl0", "cl2"]))
    assert blocks["gene_expression"].feature_names == ("a", "b")
    restored = PharmaFormerGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())
    np.testing.assert_allclose(restored.transform(features, np.array(["cl0", "cl2"])), blocks["gene_expression"].values)
