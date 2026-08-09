"""Tests for featurizer fit context."""

from __future__ import annotations

import numpy as np

from drevalpy.components.core.fitting.featurizer_fit_context import FeaturizerFitContext


def test_featurizer_fit_context_stores_training_populations() -> None:
    context = FeaturizerFitContext(
        unique_train_ids=np.array(["cl1", "cl2"], dtype=str),
        pair_expanded_train_ids=np.array(["cl1", "cl2", "cl1"], dtype=str),
        unique_early_stopping_ids=np.array(["cl2"], dtype=str),
        pair_expanded_early_stopping_ids=np.array(["cl2", "cl2"], dtype=str),
        side="cell_line",
    )
    assert context.side == "cell_line"
    np.testing.assert_array_equal(context.unique_train_ids, np.array(["cl1", "cl2"], dtype=str))
    np.testing.assert_array_equal(context.pair_expanded_train_ids, np.array(["cl1", "cl2", "cl1"], dtype=str))
