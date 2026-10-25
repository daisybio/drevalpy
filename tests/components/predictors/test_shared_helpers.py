"""Tests for the serialized-state helpers, the predictor state error and the mRMR selector."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from drevalpy._utils.torch_io import save_trusted_mapping
from drevalpy.components.predictors._mrmr import MRMRSelector, NeighborCappedKNN
from drevalpy.components.predictors._state_helpers import load_payload_state, state_float, state_mapping
from drevalpy.components.predictors.state_errors import PredictorStateError


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        pytest.param(2, 2.0, id="int"),
        pytest.param(2.5, 2.5, id="float"),
        pytest.param(np.float32(1.5), 1.5, id="numpy-scalar"),
        pytest.param("3.25", 3.25, id="numeric-string"),
        pytest.param(True, 1.0, id="bool-is-real"),
    ],
)
def test_state_float_coerces_real_and_string_values(stored: object, expected: float) -> None:
    result = state_float({"key": stored}, "key")

    assert result == pytest.approx(expected)
    assert isinstance(result, float)


def test_state_float_is_none_for_missing_or_non_numeric_values_and_raises_on_unparsable_strings() -> None:
    assert state_float({}, "dataset_mean") is None
    assert state_float({"dataset_mean": [1.0]}, "dataset_mean") is None
    with pytest.raises(ValueError):
        state_float({"dataset_mean": "not-a-number"}, "dataset_mean")


def test_state_mapping_returns_a_copy_and_empty_for_missing_or_non_mapping_values() -> None:
    stored = {"alpha": 1.0, "mode": "regression"}

    result = state_mapping({"hyperparameters": stored}, "hyperparameters")

    assert result == stored
    assert result is not stored
    assert state_mapping({}, "hyperparameters") == {}
    assert state_mapping({"hyperparameters": ["alpha"]}, "hyperparameters") == {}


def _state(payload: dict[str, object], key: str = "payload") -> dict[str, object]:
    return {key: save_trusted_mapping(payload)}


def test_load_payload_state_returns_the_decoded_payload_under_any_key() -> None:
    payload = {"predictor_hyperparameters": {"epochs": 3}, "model_state": b"weights"}

    assert load_payload_state(_state(payload), "Toy") == payload
    assert load_payload_state(_state(payload, "checkpoint"), "Toy", key="checkpoint") == payload


@pytest.mark.parametrize(
    ("state", "key", "message"),
    [
        ({}, "payload", "Toy state requires a payload byte blob"),
        ({"checkpoint": "not bytes"}, "checkpoint", "Toy state requires a checkpoint byte blob"),
        ({"payload": b"garbage"}, "payload", "Toy payload could not be deserialized"),
        (_state({"model_state": b"weights"}), "payload", "Toy payload is missing predictor_hyperparameters"),
        (_state({"predictor_hyperparameters": [1, 2]}), "payload", "missing predictor_hyperparameters"),
    ],
)
def test_load_payload_state_rejects_malformed_state(state: dict[str, object], key: str, message: str) -> None:
    with pytest.raises(PredictorStateError, match=message):
        load_payload_state(state, "Toy", key=key)


def test_predictor_state_error_is_a_catchable_runtime_error_carrying_its_message() -> None:
    with pytest.raises(RuntimeError, match="state is missing a fitted estimator"):
        raise PredictorStateError("state is missing a fitted estimator")


def test_mrmr_selector_finds_driver_and_validates_input() -> None:
    """The driver comes first, constant columns are skipped, k is capped, invalid input fails loudly."""
    rng = np.random.RandomState(0)
    x = rng.normal(size=(50, 6))
    x[:, 3] = 1.0
    y = 3.0 * x[:, 2] + rng.normal(scale=0.1, size=50)
    selector = MRMRSelector(k=4).fit(x, y)
    assert selector.selected_[0] == 2
    assert 3 not in selector.selected_
    assert len(selector.selected_) == 4
    assert selector.transform(x).shape == (50, 4)
    assert len(MRMRSelector(k=50).fit(x, y).selected_) == 5
    assert list(MRMRSelector(k=2).fit(x, np.ones(50)).selected_) == list(range(6))
    with pytest.raises(ValueError, match="finite"):
        MRMRSelector(k=2).fit(x, np.append(y[:-1], np.nan))
    with pytest.raises(ValueError, match="at least 1"):
        MRMRSelector(k=0).fit(x, y)


def test_mrmr_selector_penalizes_redundancy() -> None:
    """A copy of the best column ranks behind an independent informative column."""
    rng = np.random.RandomState(0)
    a, b = rng.normal(size=(2, 200))
    x = np.column_stack([a, a + 1e-3 * rng.normal(size=200), b, rng.normal(size=200)])
    y = a + 0.8 * b
    selected = MRMRSelector(k=2).fit(x, y).selected_
    assert 2 in selected
    assert len({0, 1} & set(selected)) == 1


def test_mrmr_selector_keeps_selection_order_accepts_float32_and_clones() -> None:
    rng = np.random.RandomState(1)
    x = rng.normal(size=(40, 5))
    y = 2.0 * x[:, 4] + x[:, 1]
    selector = MRMRSelector(k=2).fit(x, y)
    np.testing.assert_array_equal(selector.transform(x), x[:, selector.selected_])
    assert selector.selected_[0] == 4
    x32 = rng.normal(size=(30, 8)).astype(np.float32)
    assert MRMRSelector(k=3).fit(x32, x32[:, 5].astype(np.float64)).selected_[0] == 5
    assert clone(MRMRSelector(k=7, floor=0.01)).get_params() == {"k": 7, "floor": 0.01}


def test_neighbor_capped_knn_caps_neighbors_at_the_row_count() -> None:
    x = np.array([[0.0], [1.0], [2.0]])
    model = NeighborCappedKNN(n_neighbors=20, weights="uniform").fit(x, np.array([1.0, 2.0, 6.0]))
    assert model.n_neighbors == 3
    np.testing.assert_allclose(model.predict(np.array([[0.5]])), [3.0])

    many = np.arange(10, dtype=float).reshape(-1, 1)
    assert NeighborCappedKNN(n_neighbors=3).fit(many, many.ravel()).n_neighbors == 3
