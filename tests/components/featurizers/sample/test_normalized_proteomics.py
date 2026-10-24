"""Tests for normalized proteomics sample featurizer."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.sample.normalized_proteomics import (
    NormalizedProteomicsSampleFeaturizer,
    log10_and_set_na,
)
from tests.components.featurizers._name_alignment import assert_aligns_by_name, random_views
from tests.components.featurizers.sample._helpers import assert_uses_precomputed_variant
from tests.conftest import MockFeatureSource


def _make_features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"proteomics": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            "cl2": {"proteomics": np.array([4.0, 5.0, 6.0], dtype=np.float32)},
            "cl3": {"proteomics": np.array([7.0, 8.0, 9.0], dtype=np.float32)},
        }
    )


def test_normalized_proteomics_fit_transform_and_state_round_trip() -> None:
    features = _make_features()
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = NormalizedProteomicsSampleFeaturizer()
    assert featurizer.output_dim == 0
    featurizer.fit(features, entity_ids=entity_ids)

    matrix = featurizer.transform(features, entity_ids)

    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32
    restored = NormalizedProteomicsSampleFeaturizer()
    restored.set_state(featurizer.get_state())
    assert restored.output_dim == featurizer.output_dim
    np.testing.assert_allclose(restored.transform(features, entity_ids), matrix)


def test_log10_replaces_infinities_with_nan() -> None:
    transformed = log10_and_set_na(np.array([[1.0, 0.0, 100.0]]))

    np.testing.assert_allclose(transformed[0, [0, 2]], [0.0, 2.0])
    assert np.isnan(transformed[0, 1])


def test_normalized_proteomics_selects_features_and_imputes_missing_values() -> None:
    thresholded = NormalizedProteomicsSampleFeaturizer(proteomics_n_features=1)
    thresholded.fit(_make_features(), entity_ids=np.array(["cl1", "cl2", "cl3"], dtype=str))
    assert thresholded.output_dim == 3

    features = MockFeatureSource(
        features={
            "cl1": {"proteomics": np.array([1.0, 0.0, 100.0], dtype=np.float32)},
            "cl2": {"proteomics": np.array([10.0, 100.0, 1000.0], dtype=np.float32)},
        }
    )
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = NormalizedProteomicsSampleFeaturizer().fit(features, entity_ids=entity_ids)

    assert not np.isnan(featurizer.transform(features, entity_ids)).any()


def test_normalized_proteomics_serves_a_precomputed_variant() -> None:
    assert_uses_precomputed_variant(NormalizedProteomicsSampleFeaturizer())


def test_normalized_proteomics_blocks_carry_feature_names() -> None:
    features = MockFeatureSource(
        features={
            "cl1": {"proteomics": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            "cl2": {"proteomics": np.array([4.0, 5.0, 6.0], dtype=np.float32)},
        },
        meta_info={"proteomics": ["p1", "p2", "p3"]},
    )
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = NormalizedProteomicsSampleFeaturizer().fit(features, entity_ids=entity_ids)

    blocks = featurizer.transform_blocks(features, entity_ids)

    assert set(blocks) == {"proteomics"}
    assert blocks["proteomics"].feature_names == ("p1", "p2", "p3")


def test_normalized_proteomics_reads_columns_by_name() -> None:
    assert_aligns_by_name(
        lambda: NormalizedProteomicsSampleFeaturizer(proteomics_n_features=4),
        random_views({"proteomics": list("abcdef")}),
        drop_view="proteomics",
    )
