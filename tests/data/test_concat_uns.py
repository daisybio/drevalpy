"""Tests for :mod:`drevalpy.data._concat_uns`."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from drevalpy.data._concat_uns import merge_uns, same
from drevalpy.errors import InvalidInputError

NAMES = ["A", "B"]


class TestSame:
    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("tpm", "tpm"),
            (3, 3.0),
            ({"x": [1, 2], "y": {"z": "a"}}, {"y": {"z": "a"}, "x": [1, 2]}),
            (np.array([1, 2]), [1, 2]),
        ],
    )
    def test_equal_values(self, first: object, second: object) -> None:
        assert same(first, second)

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("tpm", "fpkm"),
            ({"x": 1}, {"x": 1, "y": 2}),
            ({"x": [1, 2]}, {"x": [1, 3]}),
            (np.array([1, 2]), np.array([1, 2, 3])),
            ("a", [1]),
        ],
    )
    def test_different_values(self, first: object, second: object) -> None:
        assert not same(first, second)


class TestMergeUns:
    def test_agreed_keys_are_deep_copied(self) -> None:
        first = {"nested": {"k": [1]}}

        merged = merge_uns([first, {"nested": {"k": [1]}}], names=NAMES, where="x")
        merged["nested"]["k"].append(2)

        assert first == {"nested": {"k": [1]}}

    def test_disagreeing_keys_are_dropped_with_one_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            merged = merge_uns([{"a": 1, "b": 1}, {"a": 2, "b": 2, "c": 3}], names=NAMES, where="view 'v'")

        assert merged == {}
        assert caplog.text.count("concat dropped view 'v' uns keys") == 1
        assert "a, b, c" in caplog.text

    def test_a_key_recorded_by_one_input_is_not_agreed(self) -> None:
        assert merge_uns([{"a": 1}, {}], names=NAMES, where="x") == {}

    def test_skipped_keys_are_left_out_without_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            merged = merge_uns([{"a": 1}, {"a": 1}], names=NAMES, where="x", skip={"a"})

        assert merged == {}
        assert caplog.text == ""

    @pytest.mark.parametrize(
        ("uns", "error", "match"),
        [
            ([{"u": "tpm"}, {"u": "fpkm"}], ValueError, r"view 'v': uns\['u'\].*A='tpm', B='fpkm'"),
            ([{"u": "tpm"}, {}], InvalidInputError, "B=not recorded"),
            ([{"u": "x" * 200}, {"u": "y"}], ValueError, r"^.{0,199}$"),
        ],
        ids=["differing", "recorded-by-one", "long-values-shortened"],
    )
    def test_a_strict_key_that_differs_or_is_missing_raises(self, uns: list[dict], error: type, match: str) -> None:
        with pytest.raises(error, match=match):
            merge_uns(uns, names=NAMES, where="view 'v'", strict=lambda key: True)
