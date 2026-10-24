"""Tests for the shared one-hot category encoder (``_one_hot``)."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.featurizers._one_hot import OneHotCategoryEncoder


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


def test_state_round_trip_restores_the_vocabulary_and_ignores_bad_payloads() -> None:
    encoder = _fitted("lung", "skin")
    restored = OneHotCategoryEncoder()
    restored.set_state(encoder.get_state())
    skin = np.array(["skin"], dtype=str)
    assert restored.categories == ["lung", "skin"]
    np.testing.assert_allclose(restored.transform(skin), encoder.transform(skin))

    broken = OneHotCategoryEncoder()
    broken.set_state({"categories": "lung"})
    assert broken.categories == []
