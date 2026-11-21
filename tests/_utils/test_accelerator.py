"""Tests for the accelerator detection every torch component goes through."""

from __future__ import annotations

import pytest

from drevalpy._utils import accelerator
from drevalpy._utils.accelerator import (
    FORCE_CPU_ENV_VAR,
    accelerator_available,
    ensure_available,
    lightning_accelerator,
    select_device,
)
from drevalpy.errors import DrEvalError
from drevalpy.types.enums.accelerator import Accelerator

torch = pytest.importorskip("torch")


@pytest.fixture
def fake_accelerator(monkeypatch: pytest.MonkeyPatch) -> torch.device:
    """Make torch report an MPS accelerator and clear the CPU override."""
    device = torch.device("mps")
    monkeypatch.delenv(FORCE_CPU_ENV_VAR, raising=False)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: device)
    return device


def test_an_available_accelerator_is_selected(fake_accelerator: torch.device) -> None:
    assert accelerator_available()
    assert select_device() == fake_accelerator
    assert lightning_accelerator() == "auto"


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_force_cpu_overrides_an_available_accelerator(
    fake_accelerator: torch.device, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(FORCE_CPU_ENV_VAR, value)

    assert not accelerator_available()
    assert select_device() == torch.device("cpu")
    assert lightning_accelerator() == "cpu"


def test_without_an_accelerator_everything_stays_on_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FORCE_CPU_ENV_VAR, raising=False)
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)

    assert select_device() == torch.device("cpu")
    assert lightning_accelerator() == "cpu"


def test_a_required_accelerator_must_be_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "accelerator_available", lambda: False)

    with pytest.raises(DrEvalError, match="'gpuModel' requires an accelerator"):
        ensure_available(Accelerator.REQUIRED, "gpuModel")
    ensure_available(Accelerator.OPTIONAL, "cpuFine")
    ensure_available(Accelerator.NONE, "cpuFine")


def test_a_required_accelerator_passes_when_one_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accelerator, "accelerator_available", lambda: True)

    ensure_available(Accelerator.REQUIRED, "gpuModel")


@pytest.mark.parametrize(
    ("levels", "expected"),
    [
        ([], Accelerator.NONE),
        ([Accelerator.NONE, Accelerator.OPTIONAL], Accelerator.OPTIONAL),
        ([Accelerator.REQUIRED, Accelerator.OPTIONAL, Accelerator.NONE], Accelerator.REQUIRED),
    ],
)
def test_the_strongest_level_wins(levels: list[Accelerator], expected: Accelerator) -> None:
    assert Accelerator.strongest(levels) is expected
