"""Tests for MOLIR omics preprocessing."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.sample.molir_omics import MOLIROmicsFeaturizer
from tests.components.featurizers._name_alignment import assert_aligns_by_name, random_views
from tests.components.featurizers.sample._helpers import assert_uses_precomputed_variant
from tests.conftest import MockFeatureSource


def test_molir_omics_selects_expression_and_round_trips_state() -> None:
    features = MockFeatureSource(
        {
            f"cl{i}": {
                "gene_expression": np.array([i, i * 2, 1], dtype=np.float32),
                "mutations": np.array([i, 1], dtype=np.float32),
                "copy_number_variation_gistic": np.array([2, i], dtype=np.float32),
            }
            for i in range(3)
        },
        meta_info={
            "gene_expression": ["a", "b", "c"],
            "mutations": ["m1", "m2"],
            "copy_number_variation_gistic": ["c1", "c2"],
        },
    )
    featurizer = MOLIROmicsFeaturizer(n_gene_expression_features=2).fit(features, entity_ids=np.array(["cl0", "cl1"]))
    blocks = featurizer.transform_blocks(features, np.array(["cl0", "cl2"]))
    assert set(blocks) == {"gene_expression", "mutations", "copy_number_variation_gistic"}
    assert blocks["gene_expression"].values.shape == (2, 2)
    restored = MOLIROmicsFeaturizer()
    restored.set_state(featurizer.get_state())
    np.testing.assert_allclose(restored.transform(features, np.array(["cl0", "cl2"])), blocks["gene_expression"].values)


def test_molir_omics_serves_a_precomputed_variant() -> None:
    assert_uses_precomputed_variant(MOLIROmicsFeaturizer())


def test_molir_omics_output_dim_is_zero_before_fit() -> None:
    assert MOLIROmicsFeaturizer().output_dim == 0


def test_molir_omics_hyperparameter_space_exposes_the_feature_count() -> None:
    assert set(MOLIROmicsFeaturizer.get_hyperparameter_space()) == {"n_gene_expression_features"}


def test_molir_omics_falls_back_to_empty_feature_names_without_metadata() -> None:
    features = MockFeatureSource(
        {
            f"cl{i}": {
                "gene_expression": np.array([i, i * 2, 1], dtype=np.float32),
                "mutations": np.array([i, 1], dtype=np.float32),
                "copy_number_variation_gistic": np.array([2, i], dtype=np.float32),
            }
            for i in range(2)
        }
    )

    featurizer = MOLIROmicsFeaturizer(n_gene_expression_features=2).fit(features)

    assert featurizer.get_state()["selected_feature_names"] == ()


def test_molir_omics_set_state_ignores_unrelated_keys() -> None:
    featurizer = MOLIROmicsFeaturizer()

    featurizer.set_state({"scaler": None, "mask": None, "selected_feature_names": None, "feature_names": None})

    assert featurizer.output_dim == 0


def test_selection_ranks_training_variance_before_standardization() -> None:
    features = MockFeatureSource(
        {
            f"cl{i}": {
                "gene_expression": np.array(values, dtype=np.float64),
                "mutations": np.array([0.0]),
                "copy_number_variation_gistic": np.array([1.0]),
            }
            for i, values in enumerate([[0.0, 1.0], [100.0, 1.01], [1.0, 1e8]])
        },
        meta_info={"gene_expression": ["variable", "nearly_constant"]},
    )
    featurizer = MOLIROmicsFeaturizer(n_gene_expression_features=1).fit(features, entity_ids=np.array(["cl0", "cl1"]))
    assert featurizer.get_state()["selected_feature_names"] == ("variable",)
    np.testing.assert_allclose(featurizer.transform(features, np.array(["cl0", "cl1"])), [[-1.0], [1.0]])


def test_molir_omics_reads_every_view_by_name() -> None:
    views = random_views(
        {
            "gene_expression": list("abcdef"),
            "mutations": ["m1", "m2", "m3"],
            "copy_number_variation_gistic": ["c1", "c2", "c3"],
        }
    )
    for drop_view in views:
        assert_aligns_by_name(lambda: MOLIROmicsFeaturizer(n_gene_expression_features=3), views, drop_view=drop_view)
