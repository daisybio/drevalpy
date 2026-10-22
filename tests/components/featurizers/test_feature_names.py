"""Tests for the name-based column alignment shared by the dense featurizers."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers import FeatureMismatchError
from drevalpy.components.featurizers._feature_names import align_to_fitted
from tests.components.featurizers._name_alignment import mock_source, random_views

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
