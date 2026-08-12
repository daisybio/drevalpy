"""Tests for MOLIR omics preprocessing."""

from __future__ import annotations

import numpy as np

from drevalpy.components.featurizers.cell_line.molir_omics import MOLIROmicsFeaturizer
from tests.components.featurizers.cell_line._helpers import PRECOMPUTED, precomputed_source
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


def test_molir_omics_prefers_a_precomputed_variant() -> None:
    source = precomputed_source(MOLIROmicsFeaturizer)
    ids = source.identifiers
    featurizer = MOLIROmicsFeaturizer()

    featurizer.fit(source, entity_ids=ids)

    assert featurizer.output_dim == PRECOMPUTED.shape[1]
    np.testing.assert_allclose(featurizer.transform(source, ids), PRECOMPUTED)


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
