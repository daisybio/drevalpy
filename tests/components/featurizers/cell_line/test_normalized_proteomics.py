"""Tests for normalized proteomics cell-line featurizer."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.cell_line.normalized_proteomics import (
    NormalizedProteomicsCellLineFeaturizer,
    log10_and_set_na,
)
from tests.components.featurizers.cell_line._helpers import PRECOMPUTED, precomputed_source
from tests.conftest import MockFeatureSource


def _make_features() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"proteomics": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            "cl2": {"proteomics": np.array([4.0, 5.0, 6.0], dtype=np.float32)},
            "cl3": {"proteomics": np.array([7.0, 8.0, 9.0], dtype=np.float32)},
        }
    )


def test_normalized_proteomics_fit_transform() -> None:
    featurizer = NormalizedProteomicsCellLineFeaturizer()
    features = _make_features()
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer.fit(features, entity_ids=entity_ids)
    matrix = featurizer.transform(features, entity_ids)
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32


def test_log10_replaces_infinities_with_nan() -> None:
    transformed = log10_and_set_na(np.array([[1.0, 0.0, 100.0]]))

    np.testing.assert_allclose(transformed[0, 0], 0.0)
    assert np.isnan(transformed[0, 1])
    np.testing.assert_allclose(transformed[0, 2], 2.0)


def test_normalized_proteomics_hyperparameter_space_exposes_four_tunables() -> None:
    assert set(NormalizedProteomicsCellLineFeaturizer.get_hyperparameter_space()) == {
        "proteomics_feature_threshold",
        "proteomics_n_features",
        "proteomics_normalization_downshift",
        "proteomics_normalization_width",
    }


def test_normalized_proteomics_output_dim_is_zero_before_fit() -> None:
    assert NormalizedProteomicsCellLineFeaturizer().output_dim == 0


def test_normalized_proteomics_selects_thresholded_features_when_enough_are_complete() -> None:
    featurizer = NormalizedProteomicsCellLineFeaturizer(proteomics_n_features=1)
    features = _make_features()

    featurizer.fit(features, entity_ids=np.array(["cl1", "cl2", "cl3"], dtype=str))

    assert featurizer.output_dim == 3


def test_normalized_proteomics_imputes_missing_values() -> None:
    features = MockFeatureSource(
        features={
            "cl1": {"proteomics": np.array([1.0, 0.0, 100.0], dtype=np.float32)},
            "cl2": {"proteomics": np.array([10.0, 100.0, 1000.0], dtype=np.float32)},
        }
    )
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = NormalizedProteomicsCellLineFeaturizer().fit(features, entity_ids=entity_ids)

    matrix = featurizer.transform(features, entity_ids)

    assert not np.isnan(matrix).any()


def test_normalized_proteomics_prefers_a_precomputed_variant() -> None:
    source = precomputed_source(NormalizedProteomicsCellLineFeaturizer)
    ids = source.identifiers
    featurizer = NormalizedProteomicsCellLineFeaturizer()

    featurizer.fit(source, entity_ids=ids)

    assert featurizer.output_dim == PRECOMPUTED.shape[1]
    np.testing.assert_allclose(featurizer.transform(source, ids), PRECOMPUTED)


def test_normalized_proteomics_round_trips_state() -> None:
    features = _make_features()
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = NormalizedProteomicsCellLineFeaturizer().fit(features, entity_ids=entity_ids)

    restored = NormalizedProteomicsCellLineFeaturizer()
    restored.set_state(featurizer.get_state())

    assert restored.output_dim == featurizer.output_dim
    np.testing.assert_allclose(
        restored.transform(features, entity_ids),
        featurizer.transform(features, entity_ids),
    )


def test_normalized_proteomics_set_state_ignores_unrelated_keys() -> None:
    featurizer = NormalizedProteomicsCellLineFeaturizer()

    featurizer.set_state({"proteomics_transformer": None, "view": 7, "output_dim": "many"})

    assert featurizer.output_dim == 0
    assert featurizer._view == "proteomics"


def test_normalized_proteomics_blocks_carry_feature_names() -> None:
    features = MockFeatureSource(
        features={
            "cl1": {"proteomics": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
            "cl2": {"proteomics": np.array([4.0, 5.0, 6.0], dtype=np.float32)},
        },
        meta_info={"proteomics": ["p1", "p2", "p3"]},
    )
    entity_ids = np.array(["cl1", "cl2"], dtype=str)
    featurizer = NormalizedProteomicsCellLineFeaturizer().fit(features, entity_ids=entity_ids)

    blocks = featurizer.transform_blocks(features, entity_ids)

    assert set(blocks) == {"proteomics"}
    assert blocks["proteomics"].feature_names == ("p1", "p2", "p3")
