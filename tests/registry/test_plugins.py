"""Tests for entry-point plugin discovery.

``discover_plugins`` is called at import time from ``drevalpy/registry/__init__.py``,
so its module-level ``_discovered`` latch is already ``True`` by the time any test
runs. Every test here resets the latch through ``monkeypatch``, which restores the
original value on teardown, so the latch is left exactly as the rest of the suite
found it.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any

import pytest

from drevalpy.registry import _plugins

_GROUP = "drevalpy.plugins"


class _StubEntryPoint:
    """Stand-in for :class:`importlib.metadata.EntryPoint` recording ``load`` calls."""

    def __init__(self, name: str, error: Exception | None = None) -> None:
        self.name = name
        self.loaded = False
        self._error = error

    def load(self) -> Any:
        """Record the call and optionally fail the way a broken plugin would."""
        self.loaded = True
        if self._error is not None:
            raise self._error
        return object()


class _EntryPointRecorder:
    """Replacement for ``importlib.metadata.entry_points`` that records its kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.entry_points: list[_StubEntryPoint] = []

    def __call__(self, **kwargs: Any) -> list[_StubEntryPoint]:
        """Record the query and return the entry points the test installed."""
        self.calls.append(kwargs)
        return self.entry_points


@pytest.fixture
def unlatched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the one-shot discovery latch for the duration of a single test."""
    monkeypatch.setattr(_plugins, "_discovered", False)


@pytest.fixture
def entry_points(monkeypatch: pytest.MonkeyPatch) -> _EntryPointRecorder:
    """Install a recording stub over the ``importlib.metadata`` boundary."""
    recorder = _EntryPointRecorder()
    monkeypatch.setattr(importlib.metadata, "entry_points", recorder)
    return recorder


def test_import_time_discovery_has_already_run() -> None:
    assert _plugins._discovered is True


def test_discovery_queries_only_the_drevalpy_plugin_group(unlatched: None, entry_points: _EntryPointRecorder) -> None:
    _plugins.discover_plugins()

    assert entry_points.calls == [{"group": _GROUP}]


def test_discovery_loads_every_entry_point(unlatched: None, entry_points: _EntryPointRecorder) -> None:
    entry_points.entry_points.extend([_StubEntryPoint("first"), _StubEntryPoint("second")])

    _plugins.discover_plugins()

    assert [ep.loaded for ep in entry_points.entry_points] == [True, True]


def test_discovery_sets_the_latch(unlatched: None, entry_points: _EntryPointRecorder) -> None:
    _plugins.discover_plugins()

    assert _plugins._discovered is True


def test_second_call_is_a_no_op(unlatched: None, entry_points: _EntryPointRecorder) -> None:
    _plugins.discover_plugins()
    _plugins.discover_plugins()

    assert len(entry_points.calls) == 1


def test_latched_call_does_not_query_entry_points(entry_points: _EntryPointRecorder) -> None:
    _plugins.discover_plugins()

    assert entry_points.calls == []


def test_a_failing_plugin_does_not_abort_discovery(unlatched: None, entry_points: _EntryPointRecorder) -> None:
    entry_points.entry_points.extend(
        [
            _StubEntryPoint("broken", error=RuntimeError("boom")),
            _StubEntryPoint("healthy"),
        ]
    )

    _plugins.discover_plugins()

    assert entry_points.entry_points[1].loaded is True


def test_a_failing_plugin_is_reported_as_a_warning(
    unlatched: None,
    entry_points: _EntryPointRecorder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    entry_points.entry_points.append(_StubEntryPoint("broken", error=RuntimeError("boom")))

    with caplog.at_level(logging.WARNING, logger=_plugins.__name__):
        _plugins.discover_plugins()

    assert "Failed to load drevalpy plugin 'broken'" in caplog.text
