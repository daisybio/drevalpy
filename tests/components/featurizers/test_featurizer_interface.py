"""Tests for the ``Featurizer`` interface defaults and the dense helpers."""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np
import pytest

from drevalpy.components.contracts.feature_format import FeatureFormat
from drevalpy.components.featurizers._dense import FeatureMismatchError, dense_result, view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.types.data.batch.feature_block import BlockSpec, FeaturizerResult
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
        self.names = source.get_feature_names("expr")
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        return dense_result(ids, "expr", view_matrix(source, "expr", ids, self.names), self.names)


class _IdOnly(_PassThrough):
    input_views: ClassVar[tuple[str, ...]] = ()


def _source(names: tuple[str, ...] = ("g0", "g1")) -> MockFeatureSource:
    rows = {"a": [1.0, 2.0], "b": [np.nan, np.nan], "c": [5.0, 6.0]}
    return MockFeatureSource(
        {key: {"expr": np.array(row)} for key, row in rows.items()},
        {"expr": list(names)},
    )


def test_declaration_defaults() -> None:
    assert Featurizer.sides == frozenset({"sample", "drug"})
    assert Featurizer.storage_key is None
    assert _PassThrough().stored_params() is None


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

    assert result.projection == ("expr",)
    assert result.blocks["expr"].feature_names == ("g0", "g1")
    assert result.matrix.dtype == np.float32
    np.testing.assert_array_equal(result.matrix, [[6.0, 5.0], [2.0, 1.0]])
    np.testing.assert_array_equal(result.entity_ids, ids)


def test_view_matrix_raises_when_a_fitted_feature_is_absent() -> None:
    with pytest.raises(FeatureMismatchError, match="missing 1 of the 2"):
        view_matrix(_source(("g0", "g9")), "expr", np.array(["a"]), ("g0", "g1"))
