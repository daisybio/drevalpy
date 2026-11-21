"""Tests for ``usable_pairs`` and ``prepare_features``: valid mask, fit, transform, padding and order."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Self

import numpy as np
import pytest

from drevalpy.components.featurizers._dense import FeatureMismatchError, dense_result, view_matrix
from drevalpy.components.featurizers.featurizer import Featurizer
from drevalpy.models._feature_preparation import FitIds, prepare_features, usable_pairs
from drevalpy.types.data.batch.feature_block import FeaturizerResult, numeric_feature_block, ragged_feature_block
from drevalpy.types.data.batch.response_batch import ResponseBatch
from drevalpy.types.data.feature_source import FeatureSource
from tests.conftest import MockFeatureSource

_LOGGER = "drevalpy.models._feature_preparation"


def _source(*, with_view: bool = True, names: tuple[str, ...] = ("g0", "g1")) -> MockFeatureSource:
    rows = {"a": [1.0, 2.0], "b": [np.nan, np.nan], "c": [5.0, 6.0], "d": [7.0, 8.0]}
    features = {
        key: {"raw": np.array(row) * 10, **({"expr": np.array(row)} if with_view else {})} for key, row in rows.items()
    }
    meta = {"raw": list(names), **({"expr": list(names)} if with_view else {})}
    return MockFeatureSource(features, meta)


class _Counting(Featurizer):
    """Reads ``expr`` when the source holds it, else computes it from ``view``; counts every call."""

    input_views: ClassVar[tuple[str, ...]] = ("expr",)
    source_views: ClassVar[tuple[str, ...]] = ("raw",)

    def __init__(self) -> None:
        self.calls: dict[str, int] = {"fit": 0, "transform": 0, "compute": 0, "valid_mask": 0}
        self.fit_args: dict[str, Any] = {}
        self.names: tuple[str, ...] | None = None

    def _matrix(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        if source.has_view("expr"):
            return view_matrix(source, "expr", ids, self.names)
        self.calls["compute"] += 1
        return source.view("raw", ids) / 10

    def fit(
        self,
        source: FeatureSource,
        train_ids: np.ndarray,
        *,
        train_pair_ids: np.ndarray | None = None,
        es_pair_ids: np.ndarray | None = None,
    ) -> Self:
        self.calls["fit"] += 1
        self.fit_args = {"train_ids": train_ids, "train_pair_ids": train_pair_ids, "es_pair_ids": es_pair_ids}
        self.names = source.feature_names("expr") if source.has_view("expr") else None
        return self

    def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
        self.calls["transform"] += 1
        return dense_result(ids, "expr", self._matrix(source, ids), self.names or ())

    def valid_mask(self, source: FeatureSource, ids: np.ndarray) -> np.ndarray:
        self.calls["valid_mask"] += 1
        if source.has_view("expr"):
            return super().valid_mask(source, ids)
        return ~np.all(np.isnan(source.view("raw", ids)), axis=1)


def test_view_present_is_read_and_fit_computes_nothing() -> None:
    featurizer = _Counting()
    ids = np.array(["a", "c"])

    result = prepare_features(featurizer, _source(), ids, fit=FitIds(ids))

    assert featurizer.calls == {"fit": 1, "transform": 1, "compute": 0, "valid_mask": 0}
    np.testing.assert_array_equal(result.matrix, [[1.0, 2.0], [5.0, 6.0]])


def test_view_absent_is_computed_from_the_source_view() -> None:
    featurizer = _Counting()
    ids = np.array(["a", "c"])

    result = prepare_features(featurizer, _source(with_view=False), ids, fit=FitIds(ids))

    assert featurizer.calls["compute"] == 1
    np.testing.assert_allclose(result.matrix, [[1.0, 2.0], [5.0, 6.0]])


def test_present_but_invalid_view_raises_instead_of_computing() -> None:
    featurizer = _Counting()
    ids = np.array(["a", "c"])
    prepare_features(featurizer, _source(), ids, fit=FitIds(ids))

    with pytest.raises(FeatureMismatchError, match="missing 1 of the 2"):
        prepare_features(featurizer, _source(names=("g0", "g9")), ids, fit=None)
    assert featurizer.calls["compute"] == 0


def _pairs(sample_ids: list[str], drug_ids: list[str] | None = None) -> ResponseBatch:
    drugs = drug_ids if drug_ids is not None else ["x"] * len(sample_ids)
    return ResponseBatch(np.ones(len(sample_ids)), np.array(sample_ids), np.array(drugs))


def test_usable_pairs_ask_each_featurizer_once_on_the_unique_train_and_stopping_entities() -> None:
    featurizer = _Counting()
    response = _pairs(["a", "b", "a", "c"])
    stopping = _pairs(["b", "d"])

    train, es = usable_pairs(response, stopping, (("sample_ids", featurizer, _source()), ("drug_ids", None, None)))

    assert featurizer.calls["valid_mask"] == 1
    np.testing.assert_array_equal(train.sample_ids, ["a", "a", "c"])
    assert es is not None
    np.testing.assert_array_equal(es.sample_ids, ["d"])


def test_usable_pairs_drop_missing_targets_and_keep_order_without_stopping() -> None:
    response = ResponseBatch(np.array([1.0, np.nan, 2.0]), np.array(["c", "a", "a"]), np.array(["x", "y", "z"]))

    train, es = usable_pairs(response, None, (("sample_ids", _Counting(), _source()),))

    assert es is None
    np.testing.assert_array_equal(train.drug_ids, ["x", "z"])


def test_fit_trusts_the_usable_ids_and_prediction_pads_unusable_entities() -> None:
    featurizer = _Counting()
    pairs = np.array(["a", "a", "c"])

    prepare_features(featurizer, _source(), pairs, fit=FitIds(pairs, np.array(["d"])))
    result = prepare_features(featurizer, _source(), np.array(["b", "a", "c"]), fit=None)

    np.testing.assert_array_equal(featurizer.fit_args["train_ids"], ["a", "c"])
    np.testing.assert_array_equal(featurizer.fit_args["train_pair_ids"], ["a", "a", "c"])
    np.testing.assert_array_equal(featurizer.fit_args["es_pair_ids"], ["d"])
    assert featurizer.calls["valid_mask"] == 1
    np.testing.assert_array_equal(result.entity_ids, ["b", "a", "c"])
    assert np.isnan(result.matrix[0]).all()
    np.testing.assert_array_equal(result.matrix[1:], [[1.0, 2.0], [5.0, 6.0]])
    assert result.blocks["expr"].feature_names == ("g0", "g1")


def test_object_blocks_are_padded_with_none() -> None:
    class _Ragged(_Counting):
        def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
            values = np.empty(len(ids), dtype=object)
            values[:] = [np.arange(i + 1) for i in range(len(ids))]
            return FeaturizerResult(ids, {"tokens": ragged_feature_block(values)}, ())

    featurizer = _Ragged()
    prepare_features(featurizer, _source(), np.array(["a"]), fit=FitIds(np.array(["a"])))
    result = prepare_features(featurizer, _source(), np.array(["a", "b"]), fit=None)

    assert result.blocks["tokens"].values[1] is None
    np.testing.assert_array_equal(result.blocks["tokens"].values[0], [0])


@pytest.mark.parametrize(("threshold", "warns"), [(0.2, True), (0.5, False)])
def test_the_nan_threshold_warns(caplog: pytest.LogCaptureFixture, threshold: float, warns: bool) -> None:
    featurizer = type("_Counting", (_Counting,), {"nan_threshold": threshold})()
    ids = np.array(["a", "b", "c"])

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        usable_pairs(_pairs(list(ids)), None, (("sample_ids", featurizer, _source()),))
        prepare_features(featurizer, _source(), ids, fit=None)

    messages = [record.getMessage() for record in caplog.records]
    assert any("_Counting.fit: 33% of inputs are invalid (threshold:" in m for m in messages) is warns
    assert any("_Counting.transform: 33% of inputs are invalid" in m for m in messages) is warns


def test_entity_order_is_kept_and_repeats_collapse_to_first_seen_order() -> None:
    featurizer = _Counting()
    ids = np.array(["d", "a", "d", "c", "a"])

    result = prepare_features(featurizer, _source(), ids, fit=FitIds(ids))

    np.testing.assert_array_equal(result.entity_ids, ["d", "a", "c"])
    np.testing.assert_array_equal(result.matrix, [[7.0, 8.0], [1.0, 2.0], [5.0, 6.0]])


def test_a_transform_that_reorders_entities_raises() -> None:
    class _Reversing(_Counting):
        def transform(self, source: FeatureSource, ids: np.ndarray) -> FeaturizerResult:
            values = np.zeros((len(ids), 1), dtype=np.float32)
            return FeaturizerResult(ids[::-1], {"x": numeric_feature_block(values)}, ("x",))

    with pytest.raises(ValueError, match="must preserve requested entity order"):
        prepare_features(_Reversing(), _source(), np.array(["a", "c"]), fit=FitIds(np.array(["a", "c"])))


def test_transform_only_does_not_fit() -> None:
    featurizer = _Counting()
    prepare_features(featurizer, _source(), np.array(["a"]), fit=FitIds(np.array(["a"])))

    prepare_features(featurizer, _source(), np.array(["c"]), fit=None)

    assert (featurizer.calls["fit"], featurizer.calls["transform"]) == (1, 2)


def test_requires_early_stopping_without_a_usable_holdout_raises() -> None:
    featurizer = type("_EarlyStopping", (_Counting,), {"requires_early_stopping": True})()

    with pytest.raises(ValueError, match="requires a usable early-stopping holdout"):
        prepare_features(featurizer, _source(), np.array(["a"]), fit=FitIds(np.array(["a"]), np.array([], dtype=str)))


def test_a_featurizer_used_on_the_other_side_raises() -> None:
    class _DrugOnly(_Counting):
        side = "drug"

    with pytest.raises(ValueError, match="_DrugOnly is a drug featurizer and cannot be used on the sample side"):
        prepare_features(_DrugOnly(), _source(), np.array(["a"]), fit=FitIds(np.array(["a"])))
