"""Tests for PCA cell-line featurizer."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers.cell_line.pca import PCACellLineFeaturizer
from tests.components.featurizers.cell_line._helpers import PRECOMPUTED, precomputed_source
from tests.conftest import MockFeatureSource


def _make_features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.arange(6, dtype=np.float32)},
            "cl2": {"gene_expression": np.arange(6, 12, dtype=np.float32)},
            "cl3": {"gene_expression": np.arange(12, 18, dtype=np.float32)},
        }
    )


def test_pca_reduces_view_dimension() -> None:
    features = _make_features()
    featurizer = PCACellLineFeaturizer(view="gene_expression", n_components=2)
    entity_ids = np.array(["cl1", "cl2", "cl3"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (3, 2)
    assert featurizer.output_dim == 2


def test_pca_requires_explicit_view() -> None:
    with pytest.raises(ValueError, match="requires an explicit view"):
        PCACellLineFeaturizer(view="")


def test_pca_state_roundtrip() -> None:
    features = _make_features()
    featurizer = PCACellLineFeaturizer(view="gene_expression", n_components=2)
    entity_ids = np.array(["cl1", "cl2", "cl3"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    restored = PCACellLineFeaturizer(view="gene_expression", n_components=2)
    restored.set_state(featurizer.get_state())
    assert np.allclose(
        featurizer.transform(features, entity_ids),
        restored.transform(features, entity_ids),
    )


def test_pca_aligns_cross_study_features_by_name() -> None:
    training = MockFeatureSource(
        features={
            "cl1": {"methylation": np.array([1.0, 2.0, 3.0])},
            "cl2": {"methylation": np.array([4.0, 5.0, 6.0])},
        },
        meta_info={"methylation": np.array(["a", "b", "c"])},
    )
    cross_study = MockFeatureSource(
        features={"cl3": {"methylation": np.array([20.0, 30.0, 40.0])}},
        meta_info={"methylation": np.array(["b", "c", "d"])},
    )
    expected = MockFeatureSource(
        features={"cl3": {"methylation": np.array([0.0, 20.0, 30.0])}},
        meta_info={"methylation": np.array(["a", "b", "c"])},
    )
    featurizer = PCACellLineFeaturizer(view="methylation", n_components=2)
    featurizer.fit(training, entity_ids=np.array(["cl1", "cl2"]))

    assert np.allclose(
        featurizer.transform(cross_study, np.array(["cl3"])),
        featurizer.transform(expected, np.array(["cl3"])),
    )


def test_pca_prefers_a_precomputed_variant_for_matching_hyperparameters() -> None:
    source = precomputed_source(PCACellLineFeaturizer, hyperparameters={"n_components": 2})
    ids = source.identifiers
    featurizer = PCACellLineFeaturizer(view="gene_expression", n_components=2)

    featurizer.fit(source, entity_ids=ids)

    assert featurizer.output_dim == PRECOMPUTED.shape[1]
    np.testing.assert_allclose(featurizer.transform(source, ids), PRECOMPUTED)


def test_pca_ignores_a_variant_computed_under_different_hyperparameters() -> None:
    source = precomputed_source(PCACellLineFeaturizer, hyperparameters={"n_components": 2})
    ids = source.identifiers
    featurizer = PCACellLineFeaturizer(view="gene_expression", n_components=1)

    featurizer.fit(source, entity_ids=ids)

    assert featurizer.output_dim == 1


def test_pca_caps_n_components_at_the_smaller_matrix_dimension() -> None:
    features = _make_features()
    featurizer = PCACellLineFeaturizer(view="gene_expression", n_components=128)

    featurizer.fit(features, entity_ids=np.array(["cl1", "cl2", "cl3"], dtype=str))

    assert featurizer.output_dim == 3


def test_pca_hyperparameter_space_exposes_n_components() -> None:
    assert set(PCACellLineFeaturizer.get_hyperparameter_space()) == {"n_components"}


def test_pca_set_state_ignores_unrelated_keys() -> None:
    featurizer = PCACellLineFeaturizer(view="gene_expression", n_components=2)

    featurizer.set_state({"pca": None, "view": 3, "n_components": "two", "output_dim": None, "feature_names": None})

    assert featurizer._view == "gene_expression"
    assert featurizer.output_dim == 0


def test_pca_set_state_restores_feature_names() -> None:
    featurizer = PCACellLineFeaturizer(view="gene_expression", n_components=2)

    featurizer.set_state({"feature_names": ("a", "b")})

    assert featurizer._feature_names == ("a", "b")
