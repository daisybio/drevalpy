"""Tests for the shared entity-level naive predictor base."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors.abstract.block import BlockPredictor
from drevalpy.components.predictors.naive._single_entity import SingleEntityNaivePredictor
from tests.components.predictors.naive._helpers import naive_batch, one_hot

FEATURES = one_hot(["e1", "e2"], ["e1", "e2"])


class _SampleSide(SingleEntityNaivePredictor):
    pass


class _DrugSide(SingleEntityNaivePredictor):
    _feature_side = "drug"


def _batch(side: str, idx: list[int], response: list[float] | None = None, features=FEATURES):
    return naive_batch(
        response=None if response is None else np.array(response),
        **{f"{side}_features": features, f"{side}_pair_idx": np.array(idx, dtype=np.int64)},
    )


def test_base_class_reads_named_blocks_and_defaults_to_the_sample_side() -> None:
    assert issubclass(SingleEntityNaivePredictor, BlockPredictor)
    assert SingleEntityNaivePredictor.input_interface == "block"
    assert SingleEntityNaivePredictor._feature_side == "sample"


@pytest.mark.parametrize(("cls", "side"), [(_SampleSide, "sample"), (_DrugSide, "drug")])
def test_recovers_per_entity_means_and_falls_back_to_the_dataset_mean(cls, side) -> None:
    predictor = cls()
    assert not predictor.is_fitted()
    assert predictor.get_state() == {}

    predictor.fit(_batch(side, [0, 0, 1, 1], [1.0, 3.0, 5.0, 7.0]))
    preds = predictor.predict(_batch(side, [0, 1, 2], features=np.vstack([FEATURES, np.zeros((1, 2))])))

    assert predictor.is_fitted()
    np.testing.assert_allclose(preds, [2.0, 6.0, 4.0])


def test_fit_and_predict_guards() -> None:
    predictor = _SampleSide()

    with pytest.raises(ValueError, match="require response values during fit"):
        predictor.fit(naive_batch(n_pairs=2))
    with pytest.raises(RuntimeError, match="batch.response is required"):
        predictor._fit(naive_batch(n_pairs=2))
    with pytest.raises(RuntimeError, match="Call fit before predict"):
        predictor.predict(naive_batch(n_pairs=2))


def test_state_round_trip_reproduces_predictions_and_is_json_friendly() -> None:
    predictor = _SampleSide()
    batch = _batch("sample", [0, 1], [2.0, 8.0])
    predictor.fit(batch)
    state = predictor.get_state()
    restored = _SampleSide()
    restored.set_state(state)

    assert isinstance(state["dataset_mean"], float)
    assert isinstance(state["effects"], list)
    assert restored.is_fitted()
    np.testing.assert_allclose(restored.predict(batch), predictor.predict(batch))
    empty = _SampleSide()
    empty.set_state({})
    assert not empty.is_fitted()
