"""Tests for SuperFELTR omics preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drevalpy.components.featurizers.sample.superfeltr_omics import SuperFELTROmicsFeaturizer
from drevalpy.errors import InvalidInputError
from tests.components.featurizers._name_alignment import assert_aligns_by_name, random_views
from tests.components.featurizers.sample._helpers import assert_uses_precomputed_variant, use_gene_lists
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


def test_superfeltr_omics_serves_a_precomputed_variant() -> None:
    assert_uses_precomputed_variant(SuperFELTROmicsFeaturizer(), expected_blocks=("gene_expression",))


def test_superfeltr_omics_without_metadata_and_before_fit() -> None:
    assert SuperFELTROmicsFeaturizer().output_dim == 0
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
    assert featurizer.output_dim == 6


def test_expression_transform_precedes_training_only_selection() -> None:
    # Raw variance favors column 0; arcsinh variance favors column 1. A held-out
    # extreme in column 0 must not change the fitted selection.
    features = MockFeatureSource(
        {
            "train0": {
                "gene_expression": np.array([100.0, 0.0]),
                "mutations": np.array([0.0, 1.0]),
                "copy_number_variation_gistic": np.array([1.0, 2.0]),
            },
            "train1": {
                "gene_expression": np.array([110.0, 2.0]),
                "mutations": np.array([1.0, 1.0]),
                "copy_number_variation_gistic": np.array([3.0, 2.0]),
            },
            "heldout": {
                "gene_expression": np.array([1e8, 3.0]),
                "mutations": np.array([2.0, 1.0]),
                "copy_number_variation_gistic": np.array([4.0, 2.0]),
            },
        }
    )
    fitted = SuperFELTROmicsFeaturizer(n_features_per_view=1).fit(features, entity_ids=np.array(["train0", "train1"]))
    blocks = fitted.transform_blocks(features, np.array(["train0", "heldout"]))
    np.testing.assert_array_equal(fitted.get_state()["masks"]["gene_expression"], [False, True])
    np.testing.assert_allclose(blocks["gene_expression"].values[:, 0], np.arcsinh([0.0, 3.0]))
    np.testing.assert_array_equal(blocks["mutations"].values[:, 0], [0.0, 2.0])
    np.testing.assert_array_equal(blocks["copy_number_variation_gistic"].values[:, 0], [1.0, 4.0])


def test_superfeltr_omics_reads_every_view_by_name() -> None:
    views = random_views(
        {
            "gene_expression": list("abcdef"),
            "mutations": ["m1", "m2", "m3", "m4"],
            "copy_number_variation_gistic": ["c1", "c2", "c3", "c4"],
        }
    )
    for drop_view in views:
        assert_aligns_by_name(lambda: SuperFELTROmicsFeaturizer(n_features_per_view=3), views, drop_view=drop_view)


def _omics() -> MockFeatureSource:
    views = ("gene_expression", "mutations", "copy_number_variation_gistic")
    return MockFeatureSource(
        {f"cl{i}": {view: np.array([i, 1, i + 1, 3 * i], dtype=np.float32) for view in views} for i in range(3)},
        meta_info={view: [f"{view}{j}" for j in range(4)] for view in views},
    )


def test_superfeltr_omics_gene_lists_restrict_the_variance_selection_per_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"two": ["mutations3", "mutations0"]})
    ids = np.array(["cl0", "cl1", "cl2"])
    featurizer = SuperFELTROmicsFeaturizer(n_features_per_view=1, gene_lists={"mutations": "two"}).fit(
        _omics(), entity_ids=ids
    )

    blocks = featurizer.transform_blocks(_omics(), ids)

    assert blocks["mutations"].feature_names == ("mutations3",)
    assert blocks["gene_expression"].feature_names == ("gene_expression3",)
    restored = SuperFELTROmicsFeaturizer()
    restored.set_state(featurizer.get_state())
    assert restored.get_state()["gene_lists"] == {"mutations": "two"}
    np.testing.assert_allclose(restored.transform(_omics(), ids), blocks["gene_expression"].values)


def test_superfeltr_omics_gene_lists_raise_when_a_view_lacks_genes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"bad": ["gene_expression0", "nope"]})
    with pytest.raises(InvalidInputError, match="lacks 1 of 2 genes in 'bad'"):
        SuperFELTROmicsFeaturizer(gene_lists={"gene_expression": "bad"}).fit(_omics(), entity_ids=np.array(["cl0"]))


def test_superfeltr_omics_gene_lists_reject_unknown_views() -> None:
    with pytest.raises(InvalidInputError, match="unknown views"):
        SuperFELTROmicsFeaturizer(gene_lists={"proteomics": "x"})
