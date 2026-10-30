"""Tests for the dense helpers, entity-ID de-duplication and the one-hot category encoder."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from drevalpy.components.featurizers import FeatureMismatchError
from drevalpy.components.featurizers._dense import align_to_fitted, dense_result, view_matrix
from drevalpy.components.featurizers._one_hot import OneHotCategoryEncoder
from drevalpy.models._feature_preparation import unique_entity_ids
from tests.components.featurizers._helpers import mock_source, random_views

_NAMES = [f"g{i}" for i in range(8)]


def _source_and_matrix(names: list[str]):
    source = mock_source(random_views({"v": names}))
    return source, source.get_view_matrix("v", source.identifiers)


def test_feature_mismatch_error_is_a_value_error() -> None:
    assert issubclass(FeatureMismatchError, ValueError)


def test_matching_names_return_the_matrix_untouched() -> None:
    source, matrix = _source_and_matrix(_NAMES)

    assert align_to_fitted(source, "v", matrix, tuple(_NAMES)) is matrix


def test_a_side_without_names_passes_through() -> None:
    source, matrix = _source_and_matrix(_NAMES)
    unnamed = mock_source(random_views({"v": _NAMES}))
    unnamed._meta_info = {}

    assert align_to_fitted(source, "v", matrix, None) is matrix
    assert align_to_fitted(unnamed, "v", matrix, tuple(_NAMES)) is matrix


def test_columns_follow_the_fitted_order_and_extras_are_dropped() -> None:
    source, matrix = _source_and_matrix(_NAMES)
    fitted = ("g5", "g1", "g7")

    aligned = align_to_fitted(source, "v", matrix, fitted)

    np.testing.assert_array_equal(aligned, matrix[:, [5, 1, 7]])


def test_missing_names_raise_with_view_count_and_first_names() -> None:
    source, matrix = _source_and_matrix(_NAMES)
    fitted = (*_NAMES, *(f"gone{i}" for i in range(7)))

    with pytest.raises(FeatureMismatchError, match=r"view 'v' is missing 7 of the 15 features") as info:
        align_to_fitted(source, "v", matrix, fitted)

    assert "'gone0'" in str(info.value)
    assert "'gone4'" in str(info.value)
    assert "'gone5'" not in str(info.value)


def test_unique_entity_ids_keeps_first_seen_order() -> None:
    assert unique_entity_ids(np.array(["cl2", "cl1", "cl2", "cl3"], dtype=str)).tolist() == ["cl2", "cl1", "cl3"]
    assert unique_entity_ids(np.array([], dtype=str)).tolist() == []


def _fitted(*categories: str) -> OneHotCategoryEncoder:
    encoder = OneHotCategoryEncoder()
    encoder.fit_categories(np.array(categories, dtype=str))
    return encoder


def test_categories_are_sorted_flattened_and_one_hot_encoded() -> None:
    encoder = _fitted("skin", "lung", "skin")
    assert (encoder.categories, encoder.output_dim) == (["lung", "skin"], 2)

    matrix = encoder.transform(np.array(["skin", "lung"], dtype=str))

    np.testing.assert_allclose(matrix, [[0.0, 1.0], [1.0, 0.0]])
    assert matrix.dtype == np.float32
    flat = OneHotCategoryEncoder()
    flat.fit_categories(np.array([["lung"], ["skin"]], dtype=str))
    assert flat.categories == ["lung", "skin"]


def test_empty_vocabulary_and_unfitted_encoder_are_zero_width() -> None:
    assert OneHotCategoryEncoder().transform(np.array(["lung", "skin"], dtype=str)).shape == (2, 0)
    encoder = _fitted()
    assert (encoder.categories, encoder.output_dim) == ([], 0)


def test_unknown_categories_are_zero_rows_unless_disabled() -> None:
    encoder = _fitted("lung")
    unknown = np.array(["skin"], dtype=str)
    np.testing.assert_allclose(encoder.transform(unknown), [[0.0]])
    with pytest.raises(KeyError, match="Unknown category"):
        encoder.transform(unknown, unknown_zero=False)


def test_view_matrix_reads_the_fitted_columns_and_dense_result_wraps_one_block() -> None:
    source, matrix = _source_and_matrix(_NAMES)
    ids = source.identifiers[::-1]

    values = view_matrix(source, "v", ids, ("g3", "g0"))
    result = dense_result(ids, "v", values, ("g3", "g0"))

    np.testing.assert_allclose(values, matrix[::-1][:, [3, 0]])
    assert result.projection == ("v",)
    assert result.blocks["v"].feature_names == ("g3", "g0")
    assert result.matrix.dtype == np.float32


def test_a_fitted_encoder_pickles_with_its_vocabulary() -> None:
    encoder = _fitted("lung", "skin")
    restored = pickle.loads(pickle.dumps(encoder))  # noqa: S301 -- pickled in the same expression
    skin = np.array(["skin"], dtype=str)
    assert restored.categories == ["lung", "skin"]
    np.testing.assert_allclose(restored.transform(skin), encoder.transform(skin))
