"""Tests for the generic aligned-row fetch used by every feature accessor.

The behaviour under test is the alignment contract: the result always has one
row per requested id, in the requested order, with NaN standing in for ids the
source does not carry -- and ``strict`` turns that silent fill into an error.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from drevalpy.types.data.dataset_utils.aligned_fetch import _aligned_fetch

INDEX = pd.Index(["a", "b", "c"])
DATA = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def _fetch(ids: list[str], *, strict: bool = False) -> np.ndarray:
    return _aligned_fetch(INDEX, np.array(ids), DATA, strict=strict, entity_label="cell line")


class TestAlignment:
    def test_rows_follow_the_requested_order(self):
        result = _fetch(["c", "a"])

        np.testing.assert_array_equal(result, np.array([[5.0, 6.0], [1.0, 2.0]], dtype=np.float32))

    def test_a_repeated_id_yields_a_repeated_row(self):
        result = _fetch(["b", "b"])

        np.testing.assert_array_equal(result[0], result[1])

    def test_the_result_is_float32(self):
        assert _fetch(["a"]).dtype == np.float32

    def test_an_empty_request_keeps_the_feature_width(self):
        result = _fetch([])

        assert result.shape == (0, DATA.shape[1])


class TestMissingIds:
    def test_missing_ids_become_nan_rows(self):
        result = _fetch(["a", "absent"])

        assert not np.isnan(result[0]).any()
        assert np.isnan(result[1]).all()

    def test_an_all_missing_request_is_entirely_nan(self):
        result = _fetch(["absent1", "absent2"])

        assert np.isnan(result).all()

    def test_missing_ids_are_logged_once(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger="drevalpy.types.data.dataset_utils.aligned_fetch"):
            _fetch(["a", "absent"])

        assert "1 of 2 cell line IDs not found" in caplog.text

    def test_the_warning_previews_at_most_five_ids(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger="drevalpy.types.data.dataset_utils.aligned_fetch"):
            _fetch([f"absent{i}" for i in range(7)])

        assert caplog.text.count("absent") == 5

    def test_no_warning_is_logged_when_every_id_is_present(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger="drevalpy.types.data.dataset_utils.aligned_fetch"):
            _fetch(["a", "b"])

        assert caplog.text == ""


class TestStrictMode:
    def test_strict_mode_raises_for_a_missing_id(self):
        with pytest.raises(KeyError, match="1 of 2 cell line IDs not found"):
            _fetch(["a", "absent"], strict=True)

    def test_strict_mode_passes_when_every_id_is_present(self):
        result = _fetch(["a", "b"], strict=True)

        assert result.shape == (2, 2)

    def test_the_entity_label_appears_in_the_error(self):
        with pytest.raises(KeyError, match="drug IDs not found"):
            _aligned_fetch(INDEX, np.array(["absent"]), DATA, strict=True, entity_label="drug")
