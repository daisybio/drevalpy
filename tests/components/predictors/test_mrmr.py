"""Tests of the mRMR feature selection and the neighbour-capped kNN used by ``singleDrugKnn``."""

from __future__ import annotations

import numpy as np
import pytest

from drevalpy.components.predictors._mrmr import MRMRSelector, NeighborCappedKNN


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
    from sklearn.base import clone

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
