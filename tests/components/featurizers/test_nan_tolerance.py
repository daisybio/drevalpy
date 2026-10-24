"""Tests for the featurizer NaN-tolerance policy (``_nan_tolerance``)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from drevalpy.components.featurizers._nan_tolerance import NanToleranceMixin
from drevalpy.types.data.batch.feature_block import (
    metadata_feature_block,
    numeric_feature_block,
    ragged_feature_block,
)
from tests.components.featurizers._helpers import DoublingFeaturizer, StubSource

_LOGGER_NAME = "drevalpy.components.featurizers._nan_tolerance"


def _valid(matrix, ids, probe_ids=None):
    source = StubSource(np.asarray(matrix, dtype=np.float32), np.array(ids))
    return DoublingFeaturizer()._detect_valid(source, np.array(probe_ids or ids)).tolist()


def test_the_mixin_is_part_of_the_featurizer_base() -> None:
    assert issubclass(DoublingFeaturizer, NanToleranceMixin)


def test_all_nan_rows_are_invalid() -> None:
    assert _valid([[np.nan] * 2, [1.0, 2.0], [np.nan] * 2], ["A", "B", "C"]) == [False, True, False]


@pytest.mark.parametrize("attr", ["entity_id_only", "input_views"])
def test_featurizers_without_a_probeable_view_are_all_valid(monkeypatch: pytest.MonkeyPatch, attr: str) -> None:
    monkeypatch.setattr(DoublingFeaturizer, attr, True if attr == "entity_id_only" else None)
    assert _valid([[np.nan] * 3], ["A"]) == [True]


def test_an_unreadable_view_is_all_valid() -> None:
    assert _valid(np.zeros((1, 3)), ["A"], ["missing"]) == [True]


def test_a_non_numeric_view_is_all_valid() -> None:
    source = StubSource(np.array([["a", "b"]], dtype=str), np.array(["A"]))
    assert DoublingFeaturizer()._detect_valid(source, np.array(["A"])).tolist() == [True]


def test_expand_blocks_pads_numeric_with_nan_and_keeps_feature_names() -> None:
    block = numeric_feature_block(np.array([[1.0, 2.0]], dtype=np.float32), feature_names=("g1", "g2"))
    expanded = DoublingFeaturizer()._expand_blocks_with_nan({"numeric": block}, np.array([True, False]))
    values = expanded["numeric"].values
    assert values.shape == (2, 2)
    np.testing.assert_allclose(values[0], [1.0, 2.0])
    assert np.all(np.isnan(values[1]))
    assert expanded["numeric"].feature_names == ("g1", "g2")


def test_expand_blocks_leaves_metadata_untouched_and_pads_ragged_with_none() -> None:
    meta = metadata_feature_block(np.asarray(["lung", "skin"], dtype=str))
    payload = np.empty(1, dtype=object)
    payload[0] = np.ones((2, 3), dtype=np.float32)
    blocks = {"categories": meta, "ragged": ragged_feature_block(payload)}
    expanded = DoublingFeaturizer()._expand_blocks_with_nan(blocks, np.array([True, False]))
    assert expanded["categories"] is meta
    assert expanded["ragged"].values.shape == (2,)
    assert expanded["ragged"].values[1] is None


@pytest.mark.parametrize(
    ("mask", "warns"),
    [([True, False, False, False], True), ([True] * 4, False), ([], False)],
)
def test_warn_if_above_threshold(caplog: pytest.LogCaptureFixture, mask: list[bool], warns: bool) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        DoublingFeaturizer()._warn_if_above_threshold(np.array(mask, dtype=bool), "probe")
    assert any("invalid" in record.message.lower() for record in caplog.records) is warns
