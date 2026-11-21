"""Tests for the ``Featurizer`` interface defaults and the dense helpers."""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np
import pytest

from drevalpy.components.contracts.feature_format import BlockSpec, FeatureFormat
from drevalpy.components.featurizers._dense import (
    FeatureMismatchError,
    PrecomputableFeaturizer,
    dense_result,
    view_matrix,
)
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.types.data.feature_block import FeaturizerOutput
from drevalpy.types.data.feature_source import FeatureSource
from tests.conftest import MockFeatureSource


class _PassThrough(Featurizer):
    input_views: ClassVar[tuple[str, ...]] = ("expr",)

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        self.names = source.feature_names("expr")
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerOutput:
        return dense_result(ids, "expr", view_matrix(source, "expr", ids, self.names), self.names)


class _IdOnly(_PassThrough):
    input_views: ClassVar[tuple[str, ...]] = ()


def _source(names: tuple[str, ...] = ("g0", "g1")) -> MockFeatureSource:
    rows = {"a": [1.0, 2.0], "b": [np.nan, np.nan], "c": [5.0, 6.0]}
    return MockFeatureSource(
        {key: {"expr": np.array(row)} for key, row in rows.items()},
        {"expr": list(names)},
    )


def test_an_unregistered_featurizer_has_no_side() -> None:
    assert not hasattr(Featurizer, "side")


@pytest.mark.parametrize(
    ("cls", "config", "expected"),
    [
        (_PassThrough, {}, (BlockSpec("expr", FeatureFormat.NUMERIC_MATRIX),)),
        (_PassThrough, {"view": "other"}, (BlockSpec("other", FeatureFormat.NUMERIC_MATRIX),)),
        (_IdOnly, {}, ()),
    ],
)
def test_output_blocks_default_to_the_view(cls: type[Featurizer], config: dict[str, Any], expected: tuple) -> None:
    assert cls.output_blocks(config) == expected


def test_valid_mask_rejects_all_nan_rows_of_the_first_input_view() -> None:
    ids = np.array(["a", "b", "c"])
    assert _PassThrough().valid_mask(_source(), ids).tolist() == [True, False, True]
    assert _IdOnly().valid_mask(_source(), ids).all()


def test_valid_mask_accepts_everything_when_the_view_is_unreadable() -> None:
    class _Labels(_PassThrough):
        input_views: ClassVar[tuple[str, ...]] = ("missing",)

    assert _Labels().valid_mask(_source(), np.array(["a", "b"])).all()


def test_fit_transform_aligns_to_fitted_names_and_emits_one_float32_block() -> None:
    featurizer = _PassThrough().fit(_source(), np.array(["a", "c"]))
    ids = np.array(["c", "a"])

    result = featurizer.transform(_source(("g1", "g0")), ids)

    assert list(result.blocks) == ["expr"]
    assert result.blocks["expr"].feature_names == ("g0", "g1")
    assert result.matrix.dtype == np.float32
    np.testing.assert_array_equal(result.matrix, [[6.0, 5.0], [2.0, 1.0]])
    np.testing.assert_array_equal(result.entity_ids, ids)


def test_view_matrix_raises_when_a_fitted_feature_is_absent() -> None:
    with pytest.raises(FeatureMismatchError, match="missing 1 of the 2"):
        view_matrix(_source(("g0", "g9")), "expr", np.array(["a"]), ("g0", "g1"))


@pytest.mark.parametrize(
    "unreadable",
    [np.array(["x", "y", "z"]), np.array([1.0, np.nan, 3.0]), np.array([[1, 2], [3, 4], [5, 6]], dtype=object)],
    ids=["text", "one-dimensional", "object"],
)
def test_valid_mask_accepts_everything_when_the_view_is_not_a_numeric_matrix(unreadable: np.ndarray) -> None:
    class _Source(MockFeatureSource):
        def view(self, name: str, ids: np.ndarray) -> np.ndarray:
            return unreadable

    assert _PassThrough().valid_mask(_Source({}), np.array(["a", "b", "c"])).tolist() == [True, True, True]


class _RecordedSource(MockFeatureSource):
    """A source whose precomputed ``cached`` view records the settings it was computed with."""

    def __init__(self, recorded: dict[str, Any] | None, *, has_view: bool = True) -> None:
        names = {"cached": ["c0", "c1"]} if has_view else {}
        super().__init__(
            {key: {"cached": np.array(row), "raw": np.array([10.0, 20.0])} for key, row in _CACHED.items()},
            {**names, "raw": ["r0", "r1"]},
        )
        self._recorded = recorded

    def view_settings(self, name: str) -> dict[str, Any] | None:
        return self._recorded if name == "cached" else None


_CACHED = {"a": [1.0, 2.0], "b": [np.nan, np.nan]}


class _Scaled(PrecomputableFeaturizer):
    """Doubles ``raw`` unless its ``cached`` view was computed with the same scale."""

    def __init__(self, scale: int = 2) -> None:
        super().__init__("cached", {"scale": scale})
        self._scale = scale

    def compute(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        return source.view("raw", ids) * self._scale


class TestPrecomputableFeaturizer:
    IDS = np.array(["a", "b"])

    def test_it_exposes_its_view_and_a_copy_of_its_settings(self) -> None:
        featurizer = _Scaled(3)

        assert featurizer.view == "cached"
        settings = featurizer.settings
        assert settings == {"scale": 3}
        settings["scale"] = 99
        assert featurizer.settings == {"scale": 3}

    @pytest.mark.parametrize("recorded", [None, {"scale": 2}], ids=["unrecorded", "same-settings"])
    def test_a_matching_view_is_read_with_its_names_and_checked_for_missing_rows(self, recorded) -> None:
        source = _RecordedSource(recorded)
        featurizer = _Scaled().fit(source, self.IDS)

        result = featurizer.transform(source, self.IDS[:1])

        assert result.blocks["cached"].feature_names == ("c0", "c1")
        np.testing.assert_array_equal(result.matrix, [[1.0, 2.0]])
        assert featurizer.valid_mask(source, self.IDS).tolist() == [True, False]

    @pytest.mark.parametrize(
        ("recorded", "has_view"), [({"scale": 5}, True), (None, False)], ids=["other-settings", "no-view"]
    )
    def test_otherwise_it_computes_the_output_and_every_entity_is_usable(self, recorded, has_view) -> None:
        source = _RecordedSource(recorded, has_view=has_view)
        featurizer = _Scaled().fit(source, self.IDS)

        result = featurizer.transform(source, self.IDS)

        np.testing.assert_array_equal(result.matrix, [[20.0, 40.0], [20.0, 40.0]])
        assert result.blocks["cached"].feature_names is None
        assert featurizer.valid_mask(source, self.IDS).tolist() == [True, True]

    def test_the_block_class_attribute_names_the_emitted_block(self) -> None:
        class _Named(_Scaled):
            block = "scaled"

        source = _RecordedSource(None)

        assert list(_Named().fit(source, self.IDS).transform(source, self.IDS).blocks) == ["scaled"]
        assert _Named.output_blocks({}) == (BlockSpec("scaled", FeatureFormat.NUMERIC_MATRIX),)
