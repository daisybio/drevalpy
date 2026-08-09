"""Tests for the DIPK gene-expression featurizer."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizer_fit_context import FeaturizerFitContext
from drevalpy.components.featurizers.cell_line.dipk_gene_expression import (
    DIPKGeneExpressionFeaturizer,
    GeneExpressionEncoder,
)
from tests.conftest import MockFeatureSource


def test_dipk_gene_expression_round_trips_state(monkeypatch) -> None:
    features = MockFeatureSource(
        {f"cl{i}": {"gene_expression": np.arange(3, dtype=np.float32) + i} for i in range(3)},
        meta_info={"gene_expression": ["a", "b", "c"]},
    )
    monkeypatch.setattr(
        "drevalpy.components.featurizers.cell_line.dipk_gene_expression.train_gene_expession_autoencoder",
        lambda train, validation, epochs: GeneExpressionEncoder(train.shape[1]),
    )
    context = FeaturizerFitContext(
        unique_train_ids=np.array(["cl0", "cl1"]),
        pair_expanded_train_ids=np.array(["cl0", "cl1", "cl0"]),
        unique_early_stopping_ids=np.array(["cl2"]),
        pair_expanded_early_stopping_ids=np.array(["cl2", "cl2"]),
        side="cell_line",
    )
    featurizer = DIPKGeneExpressionFeaturizer(epochs_autoencoder=1).fit(features, context=context)
    matrix = featurizer.transform(features, np.array(["cl0", "cl2"]))
    assert matrix.shape == (2, 512)
    restored = DIPKGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())
    np.testing.assert_allclose(restored.transform(features, np.array(["cl0", "cl2"])), matrix)
