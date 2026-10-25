"""Tests for seed."""

from __future__ import annotations

import os
import random
import sys
from collections.abc import Iterator
from dataclasses import replace

import numpy as np
import pytest
import torch

from drevalpy._utils.seed import drug_seed, fit_seed, seed_everything
from drevalpy.types.results._identity import RunIdentity


class _FakeCuda:
    def __init__(self, *, available: bool) -> None:
        self._available = available
        self.seeds: list[int] = []

    def is_available(self) -> bool:
        return self._available

    def manual_seed_all(self, seed: int) -> None:
        self.seeds.append(seed)


class _FakeTorch:
    """Real ``torch.manual_seed`` calls the CUDA seeder itself, so the CUDA branch needs a full stand-in."""

    def __init__(self, *, cuda_available: bool) -> None:
        self.seeds: list[int] = []
        self.cuda = _FakeCuda(available=cuda_available)

    def manual_seed(self, seed: int) -> None:
        self.seeds.append(seed)


@pytest.fixture(autouse=True)
def _restore_global_rng_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)


def _draw() -> tuple:
    return random.getstate(), np.random.rand(4).tolist(), torch.rand(4).tolist()


def test_every_backend_is_reproducible_and_seed_dependent() -> None:
    seed_everything(3)
    first = _draw()
    seed_everything(3)
    assert _draw() == first
    seed_everything(2)
    other = _draw()
    assert other[0] != first[0]
    assert other[1] != first[1]
    assert other[2] != first[2]


@pytest.mark.parametrize("available", [True, False])
def test_cuda_is_seeded_only_when_available(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    fake_torch = _FakeTorch(cuda_available=available)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    seed_everything(5)
    assert fake_torch.seeds == [5]
    assert fake_torch.cuda.seeds == ([5] if available else [])


def test_seed_everything_leaves_the_hash_seed_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    seed_everything(7)
    assert "PYTHONHASHSEED" not in os.environ


IDENTITY = RunIdentity("ElasticNet", "TOYv1", "LPO", 2, "abc")


def test_fit_seed_is_a_stable_32_bit_value_ignoring_provenance() -> None:
    assert fit_seed(42, IDENTITY) == 2377201496
    assert fit_seed(42, replace(IDENTITY, provenance='{"execution":{"seed":1}}')) == fit_seed(42, IDENTITY)
    assert fit_seed(1, IDENTITY) != fit_seed(2, IDENTITY)


@pytest.mark.parametrize(
    "change",
    [
        {"model_name": "SVR"},
        {"dataset_name": "other"},
        {"split_mode": "LSO"},
        {"fold_index": 3},
        {"fold_id": "def"},
        {"randomization": ("SVRC", "gene_expression")},
        {"robustness_repeat": 0},
    ],
)
def test_every_identity_field_changes_the_fit_seed(change: dict) -> None:
    assert fit_seed(42, replace(IDENTITY, **change)) != fit_seed(42, IDENTITY)


def test_drug_seed() -> None:
    assert drug_seed(42, "Afatinib") == 3721085039
    assert drug_seed(1, "Afatinib") != drug_seed(2, "Afatinib")
    assert drug_seed(1, "Afatinib") != drug_seed(1, "Erlotinib")
    assert drug_seed(1, "23") != drug_seed(12, "3")
