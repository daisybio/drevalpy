"""Tests for entry-point plugin discovery.

``discover_plugins`` is called at import time from ``drevalpy/registry/__init__.py``,
so its module-level ``_discovered`` latch is already ``True`` by the time any test
runs. Every test here resets the latch through ``monkeypatch``, which restores the
original value on teardown, so the latch is left exactly as the rest of the suite
found it. The failure/success ledgers are process-global for the same reason and are
patched the same way.
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

    def __init__(self, name: str, error: Exception | None = None, value: str = "") -> None:
        self.name = name
        self.value = value or f"{name}_pkg:register"
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
def ledgers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the test empty loaded/failed ledgers, restored on teardown."""
    monkeypatch.setattr(_plugins, "_LOADED_PLUGINS", {})
    monkeypatch.setattr(_plugins, "_FAILED_PLUGINS", {})


@pytest.fixture
def entry_points(monkeypatch: pytest.MonkeyPatch) -> _EntryPointRecorder:
    """Install a recording stub over the ``importlib.metadata`` boundary."""
    recorder = _EntryPointRecorder()
    monkeypatch.setattr(importlib.metadata, "entry_points", recorder)
    return recorder


@pytest.fixture
def lenient(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee non-strict behaviour regardless of the ambient environment."""
    monkeypatch.delenv(_plugins.STRICT_ENV_VAR, raising=False)


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


def test_a_failing_plugin_does_not_abort_discovery(
    unlatched: None,
    ledgers: None,
    lenient: None,
    entry_points: _EntryPointRecorder,
) -> None:
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
    ledgers: None,
    lenient: None,
    entry_points: _EntryPointRecorder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    entry_points.entry_points.append(_StubEntryPoint("broken", error=RuntimeError("boom")))

    with caplog.at_level(logging.WARNING, logger=_plugins.__name__):
        _plugins.discover_plugins()

    assert "Failed to load drevalpy plugin 'broken'" in caplog.text


def test_the_warning_points_at_the_failure_ledger(
    unlatched: None,
    ledgers: None,
    lenient: None,
    entry_points: _EntryPointRecorder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    entry_points.entry_points.append(_StubEntryPoint("broken", error=RuntimeError("boom")))

    with caplog.at_level(logging.WARNING, logger=_plugins.__name__):
        _plugins.discover_plugins()

    assert "get_failed_plugins()" in caplog.text
    assert _plugins.STRICT_ENV_VAR in caplog.text


# ---------------------------------------------------------------------------
# Failure ledger
# ---------------------------------------------------------------------------


def test_a_failing_plugin_is_recorded(
    unlatched: None,
    ledgers: None,
    lenient: None,
    entry_points: _EntryPointRecorder,
) -> None:
    entry_points.entry_points.append(_StubEntryPoint("broken", error=RuntimeError("boom")))

    _plugins.discover_plugins()

    assert list(_plugins.get_failed_plugins()) == ["broken"]


def test_the_recorded_failure_is_the_traceback(
    unlatched: None,
    ledgers: None,
    lenient: None,
    entry_points: _EntryPointRecorder,
) -> None:
    entry_points.entry_points.append(_StubEntryPoint("broken", error=RuntimeError("boom")))

    _plugins.discover_plugins()

    assert "RuntimeError: boom" in _plugins.get_failed_plugins()["broken"]


def test_get_failed_plugins_returns_a_copy(
    unlatched: None,
    ledgers: None,
    lenient: None,
    entry_points: _EntryPointRecorder,
) -> None:
    entry_points.entry_points.append(_StubEntryPoint("broken", error=RuntimeError("boom")))
    _plugins.discover_plugins()

    _plugins.get_failed_plugins().clear()

    assert list(_plugins.get_failed_plugins()) == ["broken"]


def test_a_healthy_plugin_is_not_recorded_as_failed(
    unlatched: None,
    ledgers: None,
    lenient: None,
    entry_points: _EntryPointRecorder,
) -> None:
    entry_points.entry_points.append(_StubEntryPoint("healthy"))

    _plugins.discover_plugins()

    assert _plugins.get_failed_plugins() == {}


def test_a_healthy_plugin_is_recorded_with_its_entry_point_value(
    unlatched: None,
    ledgers: None,
    lenient: None,
    entry_points: _EntryPointRecorder,
) -> None:
    entry_points.entry_points.append(_StubEntryPoint("healthy", value="my_pkg.plugin:setup"))

    _plugins.discover_plugins()

    assert _plugins.get_loaded_plugins() == {"healthy": "my_pkg.plugin:setup"}


def test_get_loaded_plugins_returns_a_copy(
    unlatched: None,
    ledgers: None,
    lenient: None,
    entry_points: _EntryPointRecorder,
) -> None:
    entry_points.entry_points.append(_StubEntryPoint("healthy"))
    _plugins.discover_plugins()

    _plugins.get_loaded_plugins().clear()

    assert list(_plugins.get_loaded_plugins()) == ["healthy"]


def test_a_recovered_plugin_leaves_the_failure_ledger(
    unlatched: None,
    ledgers: None,
    lenient: None,
    entry_points: _EntryPointRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_points.entry_points.append(_StubEntryPoint("flaky", error=RuntimeError("boom")))
    _plugins.discover_plugins()
    assert "flaky" in _plugins.get_failed_plugins()

    entry_points.entry_points[:] = [_StubEntryPoint("flaky")]
    monkeypatch.setattr(_plugins, "_discovered", False)
    _plugins.discover_plugins()

    assert _plugins.get_failed_plugins() == {}


def test_a_regressed_plugin_leaves_the_loaded_ledger(
    unlatched: None,
    ledgers: None,
    lenient: None,
    entry_points: _EntryPointRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_points.entry_points.append(_StubEntryPoint("flaky"))
    _plugins.discover_plugins()

    entry_points.entry_points[:] = [_StubEntryPoint("flaky", error=RuntimeError("boom"))]
    monkeypatch.setattr(_plugins, "_discovered", False)
    _plugins.discover_plugins()

    assert _plugins.get_loaded_plugins() == {}


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_strict_mode_recognises_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(_plugins.STRICT_ENV_VAR, value)

    assert _plugins.strict_plugins_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_strict_mode_rejects_other_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(_plugins.STRICT_ENV_VAR, value)

    assert _plugins.strict_plugins_enabled() is False


def test_strict_mode_is_off_when_unset(lenient: None) -> None:
    assert _plugins.strict_plugins_enabled() is False


def test_strict_mode_re_raises_a_plugin_failure(
    unlatched: None,
    ledgers: None,
    entry_points: _EntryPointRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_plugins.STRICT_ENV_VAR, "1")
    entry_points.entry_points.append(_StubEntryPoint("broken", error=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        _plugins.discover_plugins()


def test_strict_mode_still_records_the_failure(
    unlatched: None,
    ledgers: None,
    entry_points: _EntryPointRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_plugins.STRICT_ENV_VAR, "1")
    entry_points.entry_points.append(_StubEntryPoint("broken", error=RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        _plugins.discover_plugins()

    assert "broken" in _plugins.get_failed_plugins()


def test_strict_mode_leaves_the_latch_set(
    unlatched: None,
    ledgers: None,
    entry_points: _EntryPointRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_plugins.STRICT_ENV_VAR, "1")
    entry_points.entry_points.append(_StubEntryPoint("broken", error=RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        _plugins.discover_plugins()

    assert _plugins._discovered is True


def test_strict_mode_does_not_load_plugins_after_the_failure(
    unlatched: None,
    ledgers: None,
    entry_points: _EntryPointRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_plugins.STRICT_ENV_VAR, "1")
    entry_points.entry_points.extend(
        [
            _StubEntryPoint("broken", error=RuntimeError("boom")),
            _StubEntryPoint("healthy"),
        ]
    )

    with pytest.raises(RuntimeError):
        _plugins.discover_plugins()

    assert entry_points.entry_points[1].loaded is False
