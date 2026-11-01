"""Tests for the torch device helpers and the mRMR selector."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from drevalpy.components.predictors._mrmr import MRMRSelector, NeighborCappedKNN
from drevalpy.components.predictors.literature._torch_utils import DevicePicklingMixin, select_device
from tests.components.predictors._helpers import pickle_round_trip


class _Holder(DevicePicklingMixin):
    device_modules = ("_model", "_nets")

    def __init__(self) -> None:
        import torch

        super().__init__()
        self._model = torch.nn.Linear(2, 1)
        self._nets = [torch.nn.Linear(2, 1), None]


def test_the_device_is_selected_on_construction() -> None:
    assert _Holder()._device == select_device()


def test_select_device_follows_cuda_availability(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device() == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_device() == torch.device("cuda")


def test_the_device_is_dropped_from_the_pickle_and_reselected_on_load(monkeypatch) -> None:

    torch = pytest.importorskip("torch")
    holder = _Holder()

    assert "_device" not in holder.__getstate__()
    moved: list[object] = []
    monkeypatch.setattr(torch.nn.Module, "to", lambda module, device: moved.append(device) or module)
    loaded = pickle_round_trip(holder)

    assert loaded._device == select_device()
    assert moved == [loaded._device, loaded._device]
    torch.testing.assert_close(loaded._model.weight, holder._model.weight)


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
