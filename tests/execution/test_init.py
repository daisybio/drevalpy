"""Tests for init."""

from __future__ import annotations

from drevalpy import execution


def test_surface() -> None:
    assert sorted(execution.__all__) == ["randomization", "robustness"]
    assert not hasattr(execution, "RunResult")
    assert not hasattr(execution, "TrialResult")
