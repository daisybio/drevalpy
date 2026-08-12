"""Tests for the global RNG seeding helper."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from drevalpy.utils.seed import seed_everything


class _FakeCuda:
    """Records ``manual_seed_all`` calls without needing a GPU."""

    def __init__(self, *, available: bool) -> None:
        self._available = available
        self.seeds: list[int] = []

    def is_available(self) -> bool:
        return self._available

    def manual_seed_all(self, seed: int) -> None:
        self.seeds.append(seed)


class _FakeTorch:
    """Stand-in for ``torch`` isolating the CUDA branch.

    Patching ``torch.cuda.manual_seed_all`` in place is not enough: real
    ``torch.manual_seed`` calls it itself, so the branch under test cannot be
    observed independently.
    """

    def __init__(self, *, cuda_available: bool) -> None:
        self.seeds: list[int] = []
        self.cuda = _FakeCuda(available=cuda_available)

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)


@pytest.fixture(autouse=True)
def _restore_global_rng_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Undo the process-wide state ``seed_everything`` deliberately mutates."""
    monkeypatch.setenv("PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED", "0"))
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)


def test_exports_the_hash_seed() -> None:
    seed_everything(7)

    assert os.environ["PYTHONHASHSEED"] == "7"


def test_defaults_to_forty_two() -> None:
    seed_everything()

    assert os.environ["PYTHONHASHSEED"] == "42"


def test_python_random_is_reproducible() -> None:
    seed_everything(3)
    first = random.getstate()

    seed_everything(3)

    assert random.getstate() == first


def test_numpy_legacy_random_is_reproducible() -> None:
    seed_everything(3)
    first = np.random.rand(4)

    seed_everything(3)

    np.testing.assert_array_equal(np.random.rand(4), first)


def test_torch_random_is_reproducible() -> None:
    seed_everything(3)
    first = torch.rand(4)

    seed_everything(3)

    assert torch.equal(torch.rand(4), first)


def test_distinct_seeds_produce_distinct_streams() -> None:
    seed_everything(1)
    first = np.random.rand(4)

    seed_everything(2)

    assert not np.array_equal(np.random.rand(4), first)


def test_distinct_seeds_reach_the_python_backend() -> None:
    seed_everything(1)
    first = random.getstate()

    seed_everything(2)

    assert random.getstate() != first


def test_seeds_every_backend_in_one_call() -> None:
    seed_everything(11)
    expected = (random.getstate(), np.random.rand(), torch.rand(1).item())

    seed_everything(11)

    assert (random.getstate(), np.random.rand(), torch.rand(1).item()) == expected


def test_cuda_is_seeded_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = _FakeTorch(cuda_available=True)
    monkeypatch.setattr("drevalpy.utils.seed.torch", fake_torch)

    seed_everything(5)

    assert fake_torch.cuda.seeds == [5]


def test_cuda_is_skipped_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = _FakeTorch(cuda_available=False)
    monkeypatch.setattr("drevalpy.utils.seed.torch", fake_torch)

    seed_everything(5)

    assert fake_torch.seeds == [5]
    assert fake_torch.cuda.seeds == []
