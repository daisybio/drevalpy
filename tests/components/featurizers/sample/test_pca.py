"""Tests for PCA sample featurizer."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers import FeatureMismatchError
from drevalpy.components.featurizers.sample.pca import PCASampleFeaturizer
from tests.components.featurizers._name_alignment import assert_aligns_by_name, mock_source, random_views
from tests.components.featurizers.sample._helpers import assert_uses_precomputed_variant, precomputed_source
from tests.conftest import MockFeatureSource


def _make_features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.arange(6, dtype=np.float32)},
            "cl2": {"gene_expression": np.arange(6, 12, dtype=np.float32)},
            "cl3": {"gene_expression": np.arange(12, 18, dtype=np.float32)},
        }
    )


_IDS = np.array(["cl1", "cl2", "cl3"], dtype=str)


def test_pca_reduces_view_dimension_caps_components_and_requires_a_view() -> None:
    features = _make_features()
    featurizer = PCASampleFeaturizer(view="gene_expression", n_components=2)
    featurizer.fit(features, entity_ids=_IDS)
    assert featurizer.transform(features, _IDS).shape == (3, 2)
    assert featurizer.output_dim == 2

    capped = PCASampleFeaturizer(view="gene_expression", n_components=128).fit(features, entity_ids=_IDS)
    assert capped.output_dim == 3
    with pytest.raises(ValueError, match="requires an explicit view"):
        PCASampleFeaturizer(view="")


def test_pca_state_roundtrip_restores_feature_names() -> None:
    features = _make_features()
    featurizer = PCASampleFeaturizer(view="gene_expression", n_components=2).fit(features, entity_ids=_IDS)
    restored = PCASampleFeaturizer(view="gene_expression", n_components=2)
    restored.set_state(featurizer.get_state())
    np.testing.assert_allclose(featurizer.transform(features, _IDS), restored.transform(features, _IDS))
    restored.set_state({"view_feature_names": ["a", "b"]})
    assert restored._view_names == ("a", "b")


def test_pca_reads_columns_by_name() -> None:
    assert_aligns_by_name(
        lambda: PCASampleFeaturizer(view="methylation", n_components=2),
        random_views({"methylation": list("abcdef")}),
        drop_view="methylation",
    )


def test_pca_no_longer_zero_fills_missing_features() -> None:
    training = random_views({"methylation": ["a", "b", "c"]}, n_samples=3)
    featurizer = PCASampleFeaturizer(view="methylation", n_components=2)
    source = mock_source(training)
    featurizer.fit(source, entity_ids=source.identifiers)
    cross_study = mock_source(random_views({"methylation": ["b", "c", "d"]}, n_samples=1))

    with pytest.raises(FeatureMismatchError, match=r"view 'methylation' is missing 1 of the 3 features.*'a'"):
        featurizer.transform(cross_study, cross_study.identifiers)


def test_pca_serves_a_precomputed_variant_only_for_matching_hyperparameters() -> None:
    assert_uses_precomputed_variant(
        PCASampleFeaturizer(view="gene_expression", n_components=2),
        hyperparameters={"n_components": 2},
    )
    source = precomputed_source(PCASampleFeaturizer, hyperparameters={"n_components": 2})
    featurizer = PCASampleFeaturizer(view="gene_expression", n_components=1)

    featurizer.fit(source, entity_ids=source.identifiers)

    assert featurizer.output_dim == 1
