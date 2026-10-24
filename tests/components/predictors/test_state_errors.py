"""Tests for the predictor state error type."""

from __future__ import annotations

import pytest

from drevalpy.components.predictors.state_errors import PredictorStateError


def test_predictor_state_error_is_a_catchable_runtime_error_carrying_its_message() -> None:
    with pytest.raises(RuntimeError, match="state is missing a fitted estimator"):
        raise PredictorStateError("state is missing a fitted estimator")
