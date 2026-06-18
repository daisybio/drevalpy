"""Tests for CurveCurator device selection."""

from __future__ import annotations

import os
from unittest.mock import patch

from drevalpy.curation.device import effective_device, resolve_device


def test_effective_device_uses_cpu_for_small_batches() -> None:
    assert effective_device("auto", n_curves=10, gpu_min_curves=1000) == "cpu"
    assert effective_device("cuda", n_curves=10, gpu_min_curves=1000) == "cpu"


def test_resolve_device_cuda_sets_alloc_conf(monkeypatch) -> None:
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    assert resolve_device("cuda") == "cuda"
    assert "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.backends.mps.is_available", return_value=False)
def test_resolve_device_auto_prefers_cuda(_mock_mps, _mock_cuda) -> None:
    assert resolve_device("auto") == "cuda"


@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=True)
def test_resolve_device_auto_falls_back_to_mps(_mock_mps, _mock_cuda) -> None:
    assert resolve_device("auto") == "mps"


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.backends.mps.is_available", return_value=False)
def test_effective_device_uses_accelerator_for_large_batches(_mock_mps, _mock_cuda) -> None:
    assert effective_device("auto", n_curves=5000, gpu_min_curves=1000) == "cuda"
