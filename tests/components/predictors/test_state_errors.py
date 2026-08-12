"""Tests for the predictor state error type."""

from __future__ import annotations

import pytest

from drevalpy.components.predictors.state_errors import PredictorStateError


def test_predictor_state_error_is_a_runtime_error() -> None:
    assert issubclass(PredictorStateError, RuntimeError)


def test_predictor_state_error_carries_its_message() -> None:
    error = PredictorStateError("state is missing a fitted estimator")

    assert str(error) == "state is missing a fitted estimator"


def test_predictor_state_error_is_catchable_as_runtime_error() -> None:
    with pytest.raises(RuntimeError):
        raise PredictorStateError("boom")
