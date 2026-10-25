"""Tests for init."""

from __future__ import annotations

from drevalpy import experiment


def test_surface() -> None:
    assert sorted(experiment.__all__) == ["randomization", "robustness"]
    assert not hasattr(experiment, "RunResult")
    assert not hasattr(experiment, "TrialResult")
