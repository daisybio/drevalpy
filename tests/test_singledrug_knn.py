"""Tests of the mRMR feature selection used by :class:`SingleDrugKNN`."""

import numpy as np
import pytest

from drevalpy.models.utils import MRMRSelector


def test_mrmr_selector_finds_driver_and_validates_input():
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


def test_mrmr_selector_penalizes_redundancy():
    """A copy of the best column ranks behind an independent informative column."""
    rng = np.random.RandomState(0)
    a, b = rng.normal(size=(2, 200))
    x = np.column_stack([a, a + 1e-3 * rng.normal(size=200), b, rng.normal(size=200)])
    y = a + 0.8 * b
    selected = MRMRSelector(k=2).fit(x, y).selected_
    assert 2 in selected
    assert len({0, 1} & set(selected)) == 1
