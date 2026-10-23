"""Tests for MOLIR omics preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drevalpy.components.featurizers.sample.molir_omics import MOLIROmicsFeaturizer
from drevalpy.errors import InvalidInputError
from tests.components.featurizers._name_alignment import assert_aligns_by_name, random_views
from tests.components.featurizers.sample._helpers import assert_uses_precomputed_variant, use_gene_lists
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


def _omics() -> MockFeatureSource:
    return MockFeatureSource(
        {
            f"cl{i}": {
                "gene_expression": np.array([i, i * 2, 1, 3 - i], dtype=np.float32),
                "mutations": np.array([i, 1, 0], dtype=np.float32),
                "copy_number_variation_gistic": np.array([2, i, 1], dtype=np.float32),
            }
            for i in range(3)
        },
        meta_info={
            "gene_expression": ["a", "b", "c", "d"],
            "mutations": ["m1", "m2", "m3"],
            "copy_number_variation_gistic": ["c1", "c2", "c3"],
        },
    )


def test_molir_omics_gene_lists_select_each_view_and_name_the_variance_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"ge": ["d", "b", "a"], "mut": ["m3", "m1"]})
    ids = np.array(["cl0", "cl1", "cl2"])
    featurizer = MOLIROmicsFeaturizer(
        n_gene_expression_features=2, gene_lists={"gene_expression": "ge", "mutations": "mut"}
    ).fit(_omics(), entity_ids=ids)

    blocks = featurizer.transform_blocks(_omics(), ids)

    assert blocks["mutations"].feature_names == ("m3", "m1")
    assert blocks["copy_number_variation_gistic"].feature_names == ("c1", "c2", "c3")
    assert set(blocks["gene_expression"].feature_names) <= {"a", "b", "d"}
    assert blocks["gene_expression"].values.shape == (3, 2)
    restored = MOLIROmicsFeaturizer()
    restored.set_state(featurizer.get_state())
    np.testing.assert_allclose(restored.transform(_omics(), ids), blocks["gene_expression"].values)
    assert restored.get_state()["gene_lists"] == {"gene_expression": "ge", "mutations": "mut"}


def test_molir_omics_gene_lists_raise_when_a_view_lacks_genes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mut": ["m1", "missing"]})
    with pytest.raises(InvalidInputError, match="view 'mutations' lacks 1 of 2 genes in 'mut'"):
        MOLIROmicsFeaturizer(gene_lists={"mutations": "mut"}).fit(_omics(), entity_ids=np.array(["cl0"]))


def test_molir_omics_gene_lists_reject_unknown_views() -> None:
    with pytest.raises(InvalidInputError, match="unknown views"):
        MOLIROmicsFeaturizer(gene_lists={"methylation": "x"})
