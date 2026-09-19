"""Tests for scaled gene-expression featurizer state."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.cell_line.scaled_gene_expression import ScaledGeneExpressionFeaturizer
from tests.components.featurizers.cell_line._helpers import assert_uses_precomputed_variant
from tests.conftest import MockFeatureSource


def test_scaled_gene_expression_output_dim_round_trips() -> None:
    features = MockFeatureSource(
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


def test_scaled_gene_expression_serves_a_precomputed_variant() -> None:
    assert_uses_precomputed_variant(
        ScaledGeneExpressionFeaturizer(),
        expected_blocks=("gene_expression",),
    )


def test_scaled_gene_expression_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        ScaledGeneExpressionFeaturizer()._transform_features(MockFeatureSource(features={}), np.array(["cl1"]))


def test_scaled_gene_expression_empty_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        ScaledGeneExpressionFeaturizer()._empty_transform_features(MockFeatureSource(features={}))


def test_scaled_gene_expression_all_invalid_rows_keep_the_fitted_schema() -> None:
    training = MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([0.0, 1.0], dtype=np.float32)},
            "cl2": {"gene_expression": np.array([1.0, 2.0], dtype=np.float32)},
        },
        meta_info={"gene_expression": ["g1", "g2"]},
    )
    invalid = MockFeatureSource(
        features={
            "cl3": {"gene_expression": np.array([np.nan, np.nan], dtype=np.float32)},
            "cl4": {"gene_expression": np.array([np.nan, np.nan], dtype=np.float32)},
        },
        meta_info={"gene_expression": ["g1", "g2"]},
    )
    featurizer = ScaledGeneExpressionFeaturizer().fit(training, entity_ids=np.array(["cl1", "cl2"]))

    result = featurizer.transform_features(invalid, np.array(["cl3", "cl4"]))

    assert result.matrix.shape == (2, 2)
    assert np.isnan(result.matrix).all()
    assert result.blocks["gene_expression"].feature_names == ("g1", "g2")
    assert np.isnan(result.blocks["gene_expression"].values).all()


def test_scaled_gene_expression_state_is_empty_before_fit() -> None:
    assert ScaledGeneExpressionFeaturizer().get_state() == {}


def test_scaled_gene_expression_set_state_ignores_unrelated_keys() -> None:
    featurizer = ScaledGeneExpressionFeaturizer()

    featurizer.set_state({"gene_expression_scaler": None, "view": 3, "output_dim": "many"})

    assert featurizer.output_dim == 0
    assert featurizer._view == "gene_expression"
