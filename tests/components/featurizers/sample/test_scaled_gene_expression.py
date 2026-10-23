"""Tests for scaled gene-expression featurizer state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from drevalpy.components.featurizers.sample.scaled_gene_expression import ScaledGeneExpressionFeaturizer, rank_transform
from drevalpy.errors import InvalidInputError
from tests.components.featurizers._name_alignment import assert_aligns_by_name, random_views
from tests.components.featurizers.sample._helpers import assert_uses_precomputed_variant, use_gene_lists
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


def test_scaled_gene_expression_reads_columns_by_name() -> None:
    assert_aligns_by_name(ScaledGeneExpressionFeaturizer, random_views({"gene_expression": list("abcdef")}))


def _four_genes() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)},
            "cl2": {"gene_expression": np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)},
        },
        meta_info={"gene_expression": ["g1", "g2", "g3", "g4"]},
    )


def test_scaled_gene_expression_gene_list_keeps_the_list_genes_in_list_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["g3", "g1"]})
    features = _four_genes()
    ids = np.array(["cl1", "cl2"])
    featurizer = ScaledGeneExpressionFeaturizer(gene_list="mine").fit(features, entity_ids=ids)

    blocks = featurizer.transform_blocks(features, ids)

    assert featurizer.output_dim == 2
    assert blocks["gene_expression"].feature_names == ("g3", "g1")
    whole = ScaledGeneExpressionFeaturizer().fit(features, entity_ids=ids).transform(features, ids)
    np.testing.assert_allclose(blocks["gene_expression"].values, whole[:, [2, 0]], rtol=1e-6)


def test_scaled_gene_expression_gene_list_round_trips_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["g2", "g4"]})
    features = _four_genes()
    ids = np.array(["cl1", "cl2"])
    featurizer = ScaledGeneExpressionFeaturizer(gene_list="mine").fit(features, entity_ids=ids)

    restored = ScaledGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())

    assert featurizer.get_state()["gene_list"] == "mine"
    assert restored._gene_list == "mine"
    np.testing.assert_allclose(restored.transform(features, ids), featurizer.transform(features, ids))


def test_scaled_gene_expression_gene_list_raises_when_genes_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["g1", "absent1", "absent2"]})
    with pytest.raises(InvalidInputError, match="view 'gene_expression' lacks 2 of 3 genes in 'mine'"):
        ScaledGeneExpressionFeaturizer(gene_list="mine").fit(_four_genes(), entity_ids=np.array(["cl1"]))


def test_scaled_gene_expression_matches_a_stored_variant_only_for_its_own_gene_list() -> None:
    assert ScaledGeneExpressionFeaturizer()._fetch_hyperparameters() is None
    assert ScaledGeneExpressionFeaturizer(gene_list="mine")._fetch_hyperparameters() == {"gene_list": "mine"}


def _rank_source() -> MockFeatureSource:
    values = {
        "cl1": [1.0, 5.0],
        "cl2": [2.0, 4.0],
        "cl3": [3.0, 3.0],
        "cl4": [4.0, 2.0],
        "new": [2.5, 10.0],
    }
    return MockFeatureSource(
        features={cl: {"gene_expression": np.array(v, dtype=np.float32)} for cl, v in values.items()},
        meta_info={"gene_expression": ["g1", "g2"]},
    )


def test_rank_transform_places_values_among_the_sorted_reference() -> None:
    reference = np.array([[1.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    matrix = np.array([[0.0, 4.0], [2.0, 7.0], [3.0, 2.0], [9.0, 5.0]])

    result = rank_transform(matrix, reference)

    np.testing.assert_array_equal(result, [[0.0, 0.5], [0.5, 1.0], [1.0, 0.0], [1.0, 1.0]])


def test_rank_transform_of_a_single_reference_row_is_zero() -> None:
    np.testing.assert_array_equal(rank_transform(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]])), [[0.0, 0.0]])


def test_rank_transform_reference_is_fit_on_the_training_samples_only() -> None:
    source = _rank_source()
    featurizer = ScaledGeneExpressionFeaturizer(transform="rank")
    featurizer.fit(source, entity_ids=np.array(["cl1", "cl2", "cl3", "cl4"]))

    reference = featurizer._rank_reference
    assert reference is not None
    np.testing.assert_array_equal(reference, [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    percentiles = rank_transform(np.array([[2.5, 10.0]]), reference)
    np.testing.assert_allclose(percentiles, [[2 / 3, 1.0]])
    expected = StandardScaler().fit(rank_transform(reference, reference)).transform(percentiles)
    np.testing.assert_allclose(featurizer.transform(source, np.array(["new"])), expected, rtol=1e-6, atol=1e-6)


def test_rank_transform_scores_a_sample_the_same_alone_or_in_company() -> None:
    source = _rank_source()
    featurizer = ScaledGeneExpressionFeaturizer(transform="rank").fit(
        source, entity_ids=np.array(["cl1", "cl2", "cl3"])
    )

    alone = featurizer.transform(source, np.array(["new"]))
    together = featurizer.transform(source, np.array(["cl1", "new", "cl4"]))

    np.testing.assert_array_equal(alone[0], together[1])


def test_rank_transform_round_trips_state() -> None:
    source = _rank_source()
    ids = np.array(["cl1", "cl2", "cl3", "cl4"])
    featurizer = ScaledGeneExpressionFeaturizer(transform="rank").fit(source, entity_ids=ids)

    restored = ScaledGeneExpressionFeaturizer()
    restored.set_state(featurizer.get_state())

    assert restored._transform == "rank"
    np.testing.assert_array_equal(restored.transform(source, ids), featurizer.transform(source, ids))


def test_rank_transform_matches_a_stored_variant_only_for_rank() -> None:
    assert ScaledGeneExpressionFeaturizer(transform="rank")._fetch_hyperparameters() == {"transform": "rank"}
    assert ScaledGeneExpressionFeaturizer(gene_list="mine", transform="rank")._fetch_hyperparameters() == {
        "gene_list": "mine",
        "transform": "rank",
    }


def test_unknown_transform_is_rejected() -> None:
    with pytest.raises(InvalidInputError, match="Unknown transform 'zscore'"):
        ScaledGeneExpressionFeaturizer(transform="zscore")


def test_rank_values_without_a_fitted_reference_raise() -> None:
    with pytest.raises(RuntimeError, match="no fitted rank reference"):
        ScaledGeneExpressionFeaturizer(transform="rank")._transform_values(np.ones((1, 2)))
