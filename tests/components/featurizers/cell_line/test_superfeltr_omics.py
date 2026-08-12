"""Tests for SuperFELTR omics preprocessing."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.cell_line.superfeltr_omics import SuperFELTROmicsFeaturizer
from tests.components.featurizers.cell_line._helpers import PRECOMPUTED, precomputed_source
from tests.conftest import MockFeatureSource


def test_superfeltr_omics_selects_each_view_and_round_trips_state() -> None:
    features = MockFeatureSource(
        {
            f"cl{i}": {
                "gene_expression": np.array([i, i * 2, 1], dtype=np.float32),
                "mutations": np.array([i, 1, i + 1], dtype=np.float32),
                "copy_number_variation_gistic": np.array([2, i, i + 2], dtype=np.float32),
            }
            for i in range(3)
        },
        meta_info={
            view: [f"{view}{i}" for i in range(3)]
            for view in (
                "gene_expression",
                "mutations",
                "copy_number_variation_gistic",
            )
        },
    )
    featurizer = SuperFELTROmicsFeaturizer(n_features_per_view=2).fit(features, entity_ids=np.array(["cl0", "cl1"]))
    blocks = featurizer.transform_blocks(features, np.array(["cl0", "cl2"]))
    assert all(block.values.shape == (2, 2) for block in blocks.values())
    restored = SuperFELTROmicsFeaturizer()
    restored.set_state(featurizer.get_state())
    np.testing.assert_allclose(restored.transform(features, np.array(["cl0", "cl2"])), blocks["gene_expression"].values)


def test_superfeltr_omics_prefers_a_precomputed_variant() -> None:
    source = precomputed_source(SuperFELTROmicsFeaturizer)
    ids = source.identifiers
    featurizer = SuperFELTROmicsFeaturizer()

    featurizer.fit(source, entity_ids=ids)

    np.testing.assert_allclose(featurizer.transform(source, ids), PRECOMPUTED)


def test_superfeltr_omics_blocks_use_the_precomputed_variant() -> None:
    source = precomputed_source(SuperFELTROmicsFeaturizer)
    ids = source.identifiers
    featurizer = SuperFELTROmicsFeaturizer().fit(source, entity_ids=ids)

    blocks = featurizer.transform_blocks(source, ids)

    np.testing.assert_allclose(blocks["gene_expression"].values, PRECOMPUTED)


def test_superfeltr_omics_falls_back_to_empty_feature_names_without_metadata() -> None:
    features = MockFeatureSource(
        {
            f"cl{i}": {
                "gene_expression": np.array([i, i * 2, 1], dtype=np.float32),
                "mutations": np.array([i, 1, i + 1], dtype=np.float32),
                "copy_number_variation_gistic": np.array([2, i, i + 2], dtype=np.float32),
            }
            for i in range(2)
        }
    )

    featurizer = SuperFELTROmicsFeaturizer(n_features_per_view=2).fit(features)

    assert featurizer.get_state()["feature_names"]["gene_expression"] == ()


def test_superfeltr_omics_output_dim_sums_selected_features_across_views() -> None:
    features = MockFeatureSource(
        {
            f"cl{i}": {
                "gene_expression": np.array([i, i * 2, 1], dtype=np.float32),
                "mutations": np.array([i, 1, i + 1], dtype=np.float32),
                "copy_number_variation_gistic": np.array([2, i, i + 2], dtype=np.float32),
            }
            for i in range(2)
        }
    )

    featurizer = SuperFELTROmicsFeaturizer(n_features_per_view=2).fit(features)

    assert featurizer.output_dim == 6


def test_superfeltr_omics_output_dim_is_zero_before_fit() -> None:
    assert SuperFELTROmicsFeaturizer().output_dim == 0


def test_superfeltr_omics_hyperparameter_space_exposes_the_feature_count() -> None:
    assert set(SuperFELTROmicsFeaturizer.get_hyperparameter_space()) == {"n_features_per_view"}
