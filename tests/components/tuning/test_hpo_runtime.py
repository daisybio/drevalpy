"""Tests for Ray HPO runtime helpers."""

from __future__ import annotations

from drevalpy.components.tuning.config import HPOConfig
from drevalpy.components.tuning.hpo import _select_best_result


class _FakeResult:
    def __init__(self, metrics: dict[str, float], config: dict[str, object] | None) -> None:
        self.metrics = metrics
        self.config = config


class _FakeResults:
    def __init__(self, trials: list[_FakeResult], *, raise_on_best: bool = False) -> None:
        self._trials = trials
        self._raise_on_best = raise_on_best

    def get_best_result(self, *, metric: str, mode: str) -> _FakeResult:
        _ = metric, mode
        if self._raise_on_best:
            raise RuntimeError("ray best failed")
        return self._trials[0]

    def __iter__(self):
        return iter(self._trials)


def test_select_best_result_prefers_ray_when_usable() -> None:
    cfg = HPOConfig.from_metric("RMSE")
    results = _FakeResults(
        [
            _FakeResult({"RMSE": 1.0}, {"alpha": 0.1}),
            _FakeResult({"RMSE": 0.5}, {"alpha": 0.2}),
        ]
    )
    best = _select_best_result(results, cfg)
    assert best is not None
    assert best.config == {"alpha": 0.1}


def test_select_best_result_scans_when_ray_best_is_nan() -> None:
    cfg = HPOConfig.from_metric("RMSE")
    results = _FakeResults(
        [
            _FakeResult({"RMSE": float("nan")}, {"alpha": 0.1}),
            _FakeResult({"RMSE": 0.4}, {"alpha": 0.2}),
        ]
    )
    best = _select_best_result(results, cfg)
    assert best is not None
    assert best.config == {"alpha": 0.2}


def test_select_best_result_max_mode_picks_higher_score() -> None:
    cfg = HPOConfig.from_metric("Pearson", n_trials=2)
    results = _FakeResults([], raise_on_best=True)
    results._trials = [
        _FakeResult({"Pearson": 0.2}, {"a": 1}),
        _FakeResult({"Pearson": 0.9}, {"a": 2}),
    ]
    best = _select_best_result(results, cfg)
    assert best is not None
    assert best.config == {"a": 2}


def test_select_best_result_returns_none_when_all_invalid() -> None:
    cfg = HPOConfig.from_metric("RMSE")
    results = _FakeResults([], raise_on_best=True)
    results._trials = [
        _FakeResult({"RMSE": float("nan")}, {"a": 1}),
        _FakeResult({"RMSE": 1.0}, None),
    ]
    assert _select_best_result(results, cfg) is None
