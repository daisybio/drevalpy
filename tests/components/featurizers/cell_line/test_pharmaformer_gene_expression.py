"""Tests for PharmaFormer gene preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.cell_line.pharmaformer_gene_expression import (
    PharmaFormerGeneExpressionFeaturizer,
)
from tests.components.featurizers.cell_line._helpers import assert_uses_precomputed_variant
from tests.conftest import MockFeatureSource


def _features() -> MockFeatureSource:
    return MockFeatureSource(
        {f"cl{i}": {"gene_expression": np.array([i, i + 2], dtype=np.float32)} for i in range(3)},
        meta_info={"gene_expression": ["a", "b"]},
    )


def test_pharmaformer_gene_expression_round_trips_state() -> None:
    features = _features()
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


def test_pharmaformer_gene_expression_requires_pair_expanded_ids() -> None:
    with pytest.raises(ValueError, match="requires pair_expanded_ids"):
        PharmaFormerGeneExpressionFeaturizer().fit(_features())


def test_pharmaformer_gene_expression_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        PharmaFormerGeneExpressionFeaturizer()._transform(_features(), np.array(["cl0"]))


def test_pharmaformer_gene_expression_state_is_empty_before_fit() -> None:
    assert PharmaFormerGeneExpressionFeaturizer().get_state() == {}


def test_pharmaformer_gene_expression_output_dim_is_zero_before_fit() -> None:
    assert PharmaFormerGeneExpressionFeaturizer().output_dim == 0


def test_pharmaformer_gene_expression_serves_a_precomputed_variant() -> None:
    assert_uses_precomputed_variant(
        PharmaFormerGeneExpressionFeaturizer(),
        ids_kwarg="pair_expanded_ids",
    )
