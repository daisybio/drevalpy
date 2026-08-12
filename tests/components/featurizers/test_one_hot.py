"""Tests for the shared one-hot category encoder.

Mirrors :mod:`drevalpy.components.featurizers._one_hot`, used by the identity and
tissue featurizers.
"""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers._one_hot import OneHotCategoryEncoder


def test_categories_are_sorted_and_indexed_in_order() -> None:
    encoder = OneHotCategoryEncoder()

    encoder.fit_categories(np.array(["skin", "lung", "skin"], dtype=str))

    assert encoder.categories == ["lung", "skin"]
    assert encoder.output_dim == 2


def test_transform_emits_one_hot_rows() -> None:
    encoder = OneHotCategoryEncoder()
    encoder.fit_categories(np.array(["lung", "skin"], dtype=str))

    matrix = encoder.transform(np.array(["skin", "lung"], dtype=str))

    np.testing.assert_allclose(matrix, [[0.0, 1.0], [1.0, 0.0]])
    assert matrix.dtype == np.float32


def test_transform_before_fit_returns_a_zero_width_matrix() -> None:
    encoder = OneHotCategoryEncoder()

    matrix = encoder.transform(np.array(["lung", "skin"], dtype=str))

    assert matrix.shape == (2, 0)


def test_empty_vocabulary_has_no_categories() -> None:
    encoder = OneHotCategoryEncoder()

    encoder.fit_categories(np.array([], dtype=str))

    assert encoder.categories == []
    assert encoder.output_dim == 0


def test_unknown_categories_default_to_an_all_zero_row() -> None:
    encoder = OneHotCategoryEncoder()
    encoder.fit_categories(np.array(["lung"], dtype=str))

    matrix = encoder.transform(np.array(["skin"], dtype=str))

    np.testing.assert_allclose(matrix, [[0.0]])


def test_unknown_categories_raise_when_unknown_zero_is_disabled() -> None:
    encoder = OneHotCategoryEncoder()
    encoder.fit_categories(np.array(["lung"], dtype=str))

    with pytest.raises(KeyError, match="Unknown category"):
        encoder.transform(np.array(["skin"], dtype=str), unknown_zero=False)


def test_state_round_trip_restores_the_vocabulary() -> None:
    encoder = OneHotCategoryEncoder()
    encoder.fit_categories(np.array(["lung", "skin"], dtype=str))

    restored = OneHotCategoryEncoder()
    restored.set_state(encoder.get_state())

    assert restored.categories == ["lung", "skin"]
    np.testing.assert_allclose(
        restored.transform(np.array(["skin"], dtype=str)),
        encoder.transform(np.array(["skin"], dtype=str)),
    )


def test_set_state_ignores_a_non_list_categories_payload() -> None:
    encoder = OneHotCategoryEncoder()

    encoder.set_state({"categories": "lung"})

    assert encoder.categories == []


def test_fit_categories_flattens_multi_dimensional_input() -> None:
    encoder = OneHotCategoryEncoder()

    encoder.fit_categories(np.array([["lung"], ["skin"]], dtype=str))

    assert encoder.categories == ["lung", "skin"]
