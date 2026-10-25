"""Selection of a response transformation by its pipeline option name."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from drevalpy.errors import UnknownNameError
from drevalpy.models import get_response_transformation
from drevalpy.models.response_transformation import GroupMeanCenterer


@pytest.mark.parametrize("option", ["None", "none", "NONE", None])
def test_no_transformation_requested_returns_none(option: str | None) -> None:
    assert get_response_transformation(option) is None


@pytest.mark.parametrize(
    ("option", "expected"), [("standard", StandardScaler), ("minmax", MinMaxScaler), ("robust", RobustScaler)]
)
def test_known_options_return_a_fresh_unfitted_scaler(option: str, expected: type) -> None:
    first, second = get_response_transformation(option), get_response_transformation(option)
    assert isinstance(first, expected)
    assert first is not second
    assert not hasattr(first, "mean_")
    scaled = get_response_transformation("minmax").fit_transform(np.array([[0.0], [5.0], [10.0]]))
    np.testing.assert_allclose(scaled.ravel(), [0.0, 0.5, 1.0])


def test_unknown_option_is_rejected_with_the_valid_options() -> None:
    with pytest.raises(UnknownNameError, match="Unknown response transformation quantile"):
        get_response_transformation("quantile")
    with pytest.raises(ValueError, match="'None', 'standard', 'minmax', 'robust', 'drug_mean', 'drug_tissue_mean'"):
        get_response_transformation("Standard")


@pytest.mark.parametrize(("option", "levels"), [("drug_mean", ("drug",)), ("drug_tissue_mean", ("drug", "tissue"))])
def test_group_mean_options(option: str, levels: tuple[str, ...]) -> None:
    first, second = get_response_transformation(option), get_response_transformation(option)
    assert isinstance(first, GroupMeanCenterer)
    assert first.levels == levels
    assert first is not second
