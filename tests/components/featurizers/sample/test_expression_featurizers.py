"""Tests for the gene-expression featurizers built on feature selection and scaling: scaled and landmark genes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from upath import UPath

from drevalpy.components.featurizers.sample import gene_lists
from drevalpy.components.featurizers.sample.gene_lists import resolve_gene_list
from drevalpy.components.featurizers.sample.landmark import LandmarkGenesFeaturizer, LandmarkGenesReducedFeaturizer
from drevalpy.components.featurizers.sample.scaled_gene_expression import ScaledGeneExpressionFeaturizer, rank_transform
from drevalpy.errors import InvalidInputError
from tests.components.featurizers._helpers import mock_source, random_views
from tests.components.featurizers.sample._helpers import assert_reads_by_name, pickled, use_gene_lists
from tests.conftest import MockFeatureSource


def test_scaled_gene_expression_pickle_round_trips() -> None:
    features = MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([0.0, 1.0, 2.0], dtype=np.float32)},
            "cl2": {"gene_expression": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
        },
        meta_info={"gene_expression": ["g1", "g2", "g3"]},
    )
    ids = np.array(["cl1", "cl2"])
    featurizer = ScaledGeneExpressionFeaturizer()
    featurizer.fit(features, ids)
    matrix = featurizer.transform(features, ids).matrix
    assert matrix.shape == (2, 3)

    np.testing.assert_allclose(pickled(featurizer).transform(features, ids).matrix, matrix)


def test_scaled_gene_expression_transform_before_fit_raises() -> None:
    empty = MockFeatureSource(features={})
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        ScaledGeneExpressionFeaturizer().transform(empty, np.array(["cl1"]))
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        ScaledGeneExpressionFeaturizer().transform(empty, np.array([], dtype=str))


def test_scaled_gene_expression_empty_transform_keeps_the_fitted_schema() -> None:
    training = MockFeatureSource(
        features={
            cl: {"gene_expression": np.array(v, dtype=np.float32)} for cl, v in {"cl1": [0, 1], "cl2": [1, 2]}.items()
        },
        meta_info={"gene_expression": ["g1", "g2"]},
    )
    featurizer = ScaledGeneExpressionFeaturizer().fit(training, np.array(["cl1", "cl2"]))

    result = featurizer.transform(training, np.array([], dtype=str))

    assert result.matrix.shape == (0, 2)
    assert result.blocks["gene_expression"].feature_names == ("g1", "g2")


def _four_genes() -> MockFeatureSource:
    return MockFeatureSource(
        features={
            "cl1": {"gene_expression": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)},
            "cl2": {"gene_expression": np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)},
        },
        meta_info={"gene_expression": ["g1", "g2", "g3", "g4"]},
    )


def test_scaled_gene_expression_gene_list_keeps_list_genes_in_order_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["g3", "g1"]})
    features = _four_genes()
    ids = np.array(["cl1", "cl2"])
    featurizer = ScaledGeneExpressionFeaturizer(gene_list="mine").fit(features, ids)

    blocks = featurizer.transform(features, ids).blocks

    assert blocks["gene_expression"].feature_names == ("g3", "g1")
    whole = ScaledGeneExpressionFeaturizer().fit(features, ids).transform(features, ids).matrix
    np.testing.assert_allclose(blocks["gene_expression"].values, whole[:, [2, 0]], rtol=1e-6)
    restored = pickled(featurizer)
    assert restored._gene_list == "mine"
    np.testing.assert_allclose(restored.transform(features, ids).matrix, featurizer.transform(features, ids).matrix)


def test_scaled_gene_expression_gene_list_raises_when_genes_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_gene_lists(monkeypatch, tmp_path, {"mine": ["g1", "absent1", "absent2"]})
    with pytest.raises(InvalidInputError, match="view 'gene_expression' lacks 2 of 3 genes in 'mine'"):
        ScaledGeneExpressionFeaturizer(gene_list="mine").fit(_four_genes(), np.array(["cl1"]))


def test_scaled_gene_expression_matches_a_stored_variant_only_for_its_own_options() -> None:
    assert ScaledGeneExpressionFeaturizer().stored_params() == {}
    assert ScaledGeneExpressionFeaturizer(gene_list="mine").stored_params() == {"gene_list": "mine"}
    assert ScaledGeneExpressionFeaturizer(transform="rank").stored_params() == {"transform": "rank"}
    both = ScaledGeneExpressionFeaturizer(gene_list="mine", transform="rank").stored_params()
    assert both == {"gene_list": "mine", "transform": "rank"}


def test_scaled_gene_expression_publishes_gene_expression_whatever_the_view() -> None:
    (spec,) = ScaledGeneExpressionFeaturizer.output_blocks({"view": "proteomics"})
    assert spec.name == "gene_expression"


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
    featurizer.fit(source, np.array(["cl1", "cl2", "cl3", "cl4"]))

    reference = featurizer._rank_reference
    assert reference is not None
    np.testing.assert_array_equal(reference, [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    percentiles = rank_transform(np.array([[2.5, 10.0]]), reference)
    np.testing.assert_allclose(percentiles, [[2 / 3, 1.0]])
    expected = StandardScaler().fit(rank_transform(reference, reference)).transform(percentiles)
    np.testing.assert_allclose(featurizer.transform(source, np.array(["new"])).matrix, expected, rtol=1e-6, atol=1e-6)


def test_rank_transform_scores_a_sample_the_same_alone_or_in_company() -> None:
    source = _rank_source()
    featurizer = ScaledGeneExpressionFeaturizer(transform="rank").fit(source, np.array(["cl1", "cl2", "cl3"]))

    alone = featurizer.transform(source, np.array(["new"])).matrix
    together = featurizer.transform(source, np.array(["cl1", "new", "cl4"])).matrix

    np.testing.assert_array_equal(alone[0], together[1])


def test_rank_transform_round_trips_through_pickle() -> None:
    source = _rank_source()
    ids = np.array(["cl1", "cl2", "cl3", "cl4"])
    featurizer = ScaledGeneExpressionFeaturizer(transform="rank").fit(source, ids)

    restored = pickled(featurizer)

    assert restored._transform == "rank"
    np.testing.assert_array_equal(restored.transform(source, ids).matrix, featurizer.transform(source, ids).matrix)


def test_unknown_transform_is_rejected() -> None:
    with pytest.raises(InvalidInputError, match="Unknown transform 'zscore'"):
        ScaledGeneExpressionFeaturizer(transform="zscore")


SYMBOLS = ("G1", "G2", "G3", "G4")


@pytest.fixture
def small_lists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    pd.DataFrame({"Symbol": SYMBOLS}).to_parquet(tmp_path / "landmark_genes.parquet", index=False)
    pd.DataFrame({"Symbol": SYMBOLS[:3]}).to_parquet(tmp_path / "landmark_genes_reduced.parquet", index=False)
    monkeypatch.setattr(gene_lists, "GENE_LISTS_DIR", UPath(tmp_path))
    return SYMBOLS


def _source(names: tuple[str, ...], values: dict[str, list[float]] | None = None) -> MockFeatureSource:
    values = values or {"cl1": list(range(1, len(names) + 1)), "cl2": list(range(len(names), 0, -1))}
    return MockFeatureSource(
        features={cl: {"gene_expression": np.array(row, dtype=np.float32)} for cl, row in values.items()},
        meta_info={"gene_expression": list(names)},
    )


def test_landmark_selects_the_list_in_list_order_and_persists_state(small_lists: tuple[str, ...]) -> None:
    features = _source((*reversed(small_lists), "EXTRA"), {"cl1": [1, 2, 3, 4, 5], "cl2": [5, 4, 3, 2, 1]})
    featurizer = LandmarkGenesFeaturizer(standardize=False, arcsinh=False)
    ids = np.array(["cl1", "cl2"])
    featurizer.fit(features, ids)
    matrix = featurizer.transform(features, ids).matrix
    np.testing.assert_allclose(matrix, [[4, 3, 2, 1], [2, 3, 4, 5]])

    np.testing.assert_allclose(pickled(featurizer).transform(features, ids).matrix, matrix)


def test_landmark_reduced_uses_the_reduced_list(small_lists: tuple[str, ...]) -> None:
    featurizer = LandmarkGenesReducedFeaturizer(standardize=False)
    featurizer.fit(_source(small_lists), np.array(["cl1"]))
    assert featurizer.transform(_source(small_lists), np.array(["cl1"])).matrix.shape == (1, 3)


def test_landmark_reduced_works_on_the_shipped_list() -> None:
    symbols = resolve_gene_list("landmark_genes_reduced")
    source = mock_source(random_views({"gene_expression": symbols}, n_samples=3))
    featurizer = LandmarkGenesReducedFeaturizer().fit(source, source.identifiers)
    assert featurizer.transform(source, source.identifiers).matrix.shape == (3, len(symbols))


def test_landmark_fit_errors(small_lists: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(InvalidInputError, match=r"view 'gene_expression' lacks 2 of 4 genes in 'landmark_genes'"):
        LandmarkGenesFeaturizer().fit(_source(small_lists[:2]), np.array(["cl1"]))
    unnamed = MockFeatureSource(features={"cl1": {"gene_expression": np.array([1.0, 2.0], dtype=np.float32)}})
    with pytest.raises(ValueError, match="no feature names for view"):
        LandmarkGenesFeaturizer().fit(unnamed, np.array(["cl1"]))
    pd.DataFrame({"other": ["A"]}).to_parquet(tmp_path / "landmark_genes.parquet", index=False)
    monkeypatch.setattr(gene_lists, "GENE_LISTS_DIR", UPath(tmp_path))
    with pytest.raises(ValueError, match="recognized gene-name column"):
        LandmarkGenesFeaturizer().fit(_source(("A",)), np.array(["cl1"]))


def test_landmark_scaling_options(small_lists: tuple[str, ...]) -> None:
    features = _source(small_lists, {"cl1": [1, 5, 2, 7], "cl2": [9, 2, 4, 1]})
    ids = np.array(["cl1", "cl2"])
    scaled = LandmarkGenesFeaturizer(standardize=True, minmax_scale=True).fit(features, ids)
    matrix = scaled.transform(features, ids).matrix
    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0

    raw = _source(small_lists, {"cl1": [0, 1, 2, 3]})
    arcsinh = LandmarkGenesFeaturizer(standardize=False, arcsinh=True).fit(raw, ids[:1])
    np.testing.assert_allclose(arcsinh.transform(raw, ids[:1]).matrix, np.arcsinh([[0.0, 1.0, 2.0, 3.0]]), rtol=1e-6)


def test_landmark_unfitted_featurizer_raises() -> None:
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        LandmarkGenesFeaturizer().transform(_source(SYMBOLS), np.array(["cl1"]))
    with pytest.raises(RuntimeError, match="must be fit before transform"):
        LandmarkGenesReducedFeaturizer().transform(_source(SYMBOLS), np.array(["cl1"]))


def test_landmark_blocks_carry_the_selected_gene_names_even_without_source_metadata(
    small_lists: tuple[str, ...],
) -> None:
    features = _source((*small_lists, "EXTRA"), {"cl1": [1, 2, 3, 4, 5]})
    ids = np.array(["cl1"])
    featurizer = LandmarkGenesFeaturizer(standardize=False).fit(features, ids)

    blocks = featurizer.transform(features, ids).blocks

    assert set(blocks) == {"gene_expression"}
    assert blocks["gene_expression"].feature_names == small_lists
    unnamed = MockFeatureSource(features={"cl1": {"gene_expression": np.array([1.0, 2.0, 3.0, 4.0])}})
    assert featurizer.transform(unnamed, ids).blocks["gene_expression"].feature_names == small_lists


def test_landmark_reads_genes_by_name_and_does_not_need_the_others(small_lists: tuple[str, ...]) -> None:
    fit_views = random_views({"gene_expression": [*small_lists, "NOT_A_GENE"]})
    assert_reads_by_name(LandmarkGenesFeaturizer, fit_views, drop_view="gene_expression")
    fit_source = mock_source(fit_views)
    featurizer = LandmarkGenesFeaturizer(standardize=False).fit(fit_source, fit_source.identifiers)
    without_extra = mock_source({"gene_expression": (fit_views["gene_expression"][0][:, :4], small_lists)})

    np.testing.assert_allclose(
        featurizer.transform(without_extra, without_extra.identifiers).matrix,
        featurizer.transform(fit_source, fit_source.identifiers).matrix,
    )


def _pair_fit(fit_on_pairs: bool, small_lists: tuple[str, ...], pairs: bool = True) -> LandmarkGenesFeaturizer:
    features = _source(small_lists, {"cl1": [1, 2, 3, 4], "cl2": [5, 8, 9, 20]})
    pair_ids = np.array(["cl1", "cl1", "cl1", "cl2"]) if pairs else None
    return LandmarkGenesFeaturizer(arcsinh=False, fit_on_pairs=fit_on_pairs).fit(
        features, np.array(["cl1", "cl2"]), train_pair_ids=pair_ids
    )


def test_landmark_fit_on_pairs_weights_by_pair_count_and_falls_back_without_pairs(small_lists: tuple[str, ...]) -> None:
    per_sample = [3.0, 5.0, 6.0, 12.0]
    for fit_on_pairs, pairs, mean in (
        (True, True, [2.0, 3.5, 4.5, 8.0]),
        (False, True, per_sample),
        (True, False, per_sample),
    ):
        scaler = _pair_fit(fit_on_pairs, small_lists, pairs=pairs)._scaler
        assert scaler is not None
        np.testing.assert_allclose(scaler.mean_, mean)
    assert pickled(_pair_fit(True, small_lists))._fit_on_pairs is True
