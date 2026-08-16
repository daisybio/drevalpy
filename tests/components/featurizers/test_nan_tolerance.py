"""Tests for the featurizer NaN-tolerance policy.

Mirrors :mod:`drevalpy.components.featurizers._nan_tolerance`, which holds the
three steps every public ``fit``/``transform`` on ``Featurizer`` brackets its
subclass hook with: work out which entities have usable rows, warn when too few
do, and pad the result back to full length.

The wrappers themselves stay covered in ``test_base.py``, where they are defined;
what is asserted here is the policy they delegate to, including the four
"treat it as valid" escape hatches in ``_detect_valid`` that keep a featurizer
usable against a source it cannot probe.
"""

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


def test_the_mixin_is_part_of_the_featurizer_base() -> None:
    """The policy is reached through ``Featurizer``, not wired in per subclass."""
    assert issubclass(DoublingFeaturizer, NanToleranceMixin)


class TestDetectValid:
    """Which entities the featurizer considers usable."""

    def test_all_nan_rows_are_invalid(self) -> None:
        ids = np.array(["A", "B", "C"])
        matrix = np.array([[np.nan, np.nan], [1.0, 2.0], [np.nan, np.nan]], dtype=np.float32)

        mask = DoublingFeaturizer()._detect_valid(StubSource(matrix, ids), ids)

        assert mask.tolist() == [False, True, False]

    def test_entity_id_only_featurizers_are_all_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        feat = DoublingFeaturizer()
        monkeypatch.setattr(DoublingFeaturizer, "entity_id_only", True)

        mask = feat._detect_valid(StubSource(np.zeros((1, 3)), np.array(["A"])), np.array(["A"]))

        assert mask.tolist() == [True]

    def test_a_viewless_featurizer_is_all_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        feat = DoublingFeaturizer()
        monkeypatch.setattr(DoublingFeaturizer, "input_views", None)

        mask = feat._detect_valid(StubSource(np.zeros((1, 3)), np.array(["A"])), np.array(["A"]))

        assert mask.tolist() == [True]

    def test_an_unreadable_view_is_all_valid(self) -> None:
        source = StubSource(np.zeros((1, 3)), np.array(["A"]))

        mask = DoublingFeaturizer()._detect_valid(source, np.array(["missing"]))

        assert mask.tolist() == [True]

    def test_a_non_numeric_view_is_all_valid(self) -> None:
        source = StubSource(np.array([["a", "b"]], dtype=str), np.array(["A"]))

        mask = DoublingFeaturizer()._detect_valid(source, np.array(["A"]))

        assert mask.tolist() == [True]


class TestExpandBlocksWithNan:
    """Padding valid-only blocks back out to the full entity list."""

    def test_numeric_blocks_are_padded_with_nan(self) -> None:
        block = numeric_feature_block(np.array([[1.0, 2.0]], dtype=np.float32))

        expanded = DoublingFeaturizer()._expand_blocks_with_nan({"numeric": block}, np.array([True, False]), 2)

        values = expanded["numeric"].values
        assert values.shape == (2, 2)
        np.testing.assert_allclose(values[0], [1.0, 2.0])
        assert np.all(np.isnan(values[1]))

    def test_non_entity_aligned_blocks_pass_through_untouched(self) -> None:
        block = metadata_feature_block(np.asarray(["lung", "skin"], dtype=str))

        expanded = DoublingFeaturizer()._expand_blocks_with_nan({"categories": block}, np.array([True, False]), 2)

        assert expanded["categories"] is block

    def test_ragged_payloads_are_padded_with_none(self) -> None:
        payload = np.empty(1, dtype=object)
        payload[0] = np.ones((2, 3), dtype=np.float32)

        expanded = DoublingFeaturizer()._expand_blocks_with_nan(
            {"ragged": ragged_feature_block(payload)},
            np.array([True, False]),
            2,
        )

        values = expanded["ragged"].values
        assert values.shape == (2,)
        assert values[1] is None

    def test_the_padded_block_keeps_its_feature_names(self) -> None:
        block = numeric_feature_block(np.array([[1.0]], dtype=np.float32), feature_names=("g1",))

        expanded = DoublingFeaturizer()._expand_blocks_with_nan({"numeric": block}, np.array([True, False]), 2)

        assert expanded["numeric"].feature_names == ("g1",)


class TestWarnIfAboveThreshold:
    """The warning is the only signal a run has silently lost most of its rows."""

    def test_it_warns_above_the_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            DoublingFeaturizer()._warn_if_above_threshold(np.array([True, False, False, False]), "probe")

        assert any("invalid" in record.message.lower() for record in caplog.records)

    def test_it_stays_quiet_below_the_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            DoublingFeaturizer()._warn_if_above_threshold(np.array([True, True, True, True]), "probe")

        assert not caplog.records

    def test_an_empty_mask_never_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            DoublingFeaturizer()._warn_if_above_threshold(np.array([], dtype=bool), "empty")

        assert not caplog.records
