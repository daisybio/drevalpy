"""Tests for the sklearn response-transformation lookup."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from drevalpy.utils.response_transform import get_response_transformation


@pytest.mark.parametrize(
    "option",
    [pytest.param("None", id="literal-none-string"), pytest.param(None, id="none-object")],
)
def test_no_transformation_requested_returns_none(option: str | None) -> None:
    assert get_response_transformation(option) is None


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        pytest.param("standard", StandardScaler, id="standard"),
        pytest.param("minmax", MinMaxScaler, id="minmax"),
        pytest.param("robust", RobustScaler, id="robust"),
    ],
)
def test_known_options_return_the_matching_scaler(option: str, expected: type) -> None:
    assert isinstance(get_response_transformation(option), expected)


def test_each_call_returns_a_fresh_unfitted_transformer() -> None:
    first = get_response_transformation("standard")
    second = get_response_transformation("standard")

    assert first is not second
    assert not hasattr(first, "mean_")


def test_returned_transformer_is_usable() -> None:
    transformer = get_response_transformation("minmax")

    scaled = transformer.fit_transform(np.array([[0.0], [5.0], [10.0]]))

    np.testing.assert_allclose(scaled.ravel(), [0.0, 0.5, 1.0])


def test_unknown_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown response transformation quantile"):
        get_response_transformation("quantile")


def test_rejection_message_lists_the_valid_options() -> None:
    with pytest.raises(ValueError, match="'None', 'standard', 'minmax', 'robust'"):
        get_response_transformation("Standard")
