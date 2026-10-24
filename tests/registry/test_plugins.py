"""Tests for entry-point plugin discovery.

``discover_plugins`` runs at import time, so its ``_discovered`` latch is already set
when a test starts; the fixtures reset it (and the process-global ledgers) through
``monkeypatch`` so the rest of the suite finds them as they were.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any

import pytest

from drevalpy.registry import _plugins


class _StubEntryPoint:
    def __init__(self, name: str, error: Exception | None = None, value: str = "") -> None:
        self.name = name
        self.value = value or f"{name}_pkg:register"
        self.loaded = False
        self._error = error

    def load(self) -> Any:
        self.loaded = True
        if self._error is not None:
            raise self._error
        return object()


class _EntryPointRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.entry_points: list[_StubEntryPoint] = []

    def __call__(self, **kwargs: Any) -> list[_StubEntryPoint]:
        self.calls.append(kwargs)
        return self.entry_points


@pytest.fixture
def unlatched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_plugins, "_discovered", False)


@pytest.fixture
def ledgers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_plugins, "_LOADED_PLUGINS", {})
    monkeypatch.setattr(_plugins, "_FAILED_PLUGINS", {})


@pytest.fixture
def entry_points(monkeypatch: pytest.MonkeyPatch) -> _EntryPointRecorder:
    recorder = _EntryPointRecorder()
    monkeypatch.setattr(importlib.metadata, "entry_points", recorder)
    return recorder


@pytest.fixture
def lenient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_plugins.STRICT_ENV_VAR, raising=False)


@pytest.fixture
def strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_plugins.STRICT_ENV_VAR, "1")


def _broken() -> _StubEntryPoint:
    return _StubEntryPoint("broken", error=RuntimeError("boom"))


def test_discovery_queries_the_plugin_group_loads_every_entry_point_and_latches(
    unlatched: None, ledgers: None, entry_points: _EntryPointRecorder
) -> None:
    entry_points.entry_points.extend([_StubEntryPoint("first"), _StubEntryPoint("second")])

    _plugins.discover_plugins()
    _plugins.discover_plugins()

    assert entry_points.calls == [{"group": "drevalpy.plugins"}]
    assert [ep.loaded for ep in entry_points.entry_points] == [True, True]
    assert _plugins._discovered is True


def test_latched_call_does_not_query_entry_points(entry_points: _EntryPointRecorder) -> None:
    assert _plugins._discovered is True

    _plugins.discover_plugins()

    assert entry_points.calls == []


@pytest.mark.usefixtures("unlatched", "ledgers", "lenient")
class TestLenient:
    def test_a_failing_plugin_is_warned_about_and_does_not_abort_discovery(
        self, entry_points: _EntryPointRecorder, caplog: pytest.LogCaptureFixture
    ) -> None:
        entry_points.entry_points.extend([_broken(), _StubEntryPoint("healthy", value="my_pkg.plugin:setup")])

        with caplog.at_level(logging.WARNING, logger=_plugins.__name__):
            _plugins.discover_plugins()

        assert entry_points.entry_points[1].loaded is True
        assert "Failed to load drevalpy plugin 'broken'" in caplog.text
        assert "get_failed_plugins()" in caplog.text
        assert _plugins.STRICT_ENV_VAR in caplog.text

    def test_ledgers_record_the_traceback_and_the_entry_point_value(self, entry_points: _EntryPointRecorder) -> None:
        entry_points.entry_points.extend([_broken(), _StubEntryPoint("healthy", value="my_pkg.plugin:setup")])

        _plugins.discover_plugins()

        assert list(_plugins.get_failed_plugins()) == ["broken"]
        assert "RuntimeError: boom" in _plugins.get_failed_plugins()["broken"]
        assert _plugins.get_loaded_plugins() == {"healthy": "my_pkg.plugin:setup"}

    def test_ledger_accessors_return_copies(self, entry_points: _EntryPointRecorder) -> None:
        entry_points.entry_points.extend([_broken(), _StubEntryPoint("healthy")])
        _plugins.discover_plugins()

        _plugins.get_failed_plugins().clear()
        _plugins.get_loaded_plugins().clear()

        assert list(_plugins.get_failed_plugins()) == ["broken"]
        assert list(_plugins.get_loaded_plugins()) == ["healthy"]

    @pytest.mark.parametrize("recovers", [True, False], ids=["recovered", "regressed"])
    def test_a_plugin_that_changes_state_moves_between_ledgers(
        self, entry_points: _EntryPointRecorder, monkeypatch: pytest.MonkeyPatch, recovers: bool
    ) -> None:
        healthy, failing = _StubEntryPoint("flaky"), _StubEntryPoint("flaky", error=RuntimeError("boom"))
        entry_points.entry_points.append(failing if recovers else healthy)
        _plugins.discover_plugins()

        entry_points.entry_points[:] = [healthy if recovers else failing]
        monkeypatch.setattr(_plugins, "_discovered", False)
        _plugins.discover_plugins()

        gone, present = (_plugins.get_failed_plugins(), _plugins.get_loaded_plugins())[:: 1 if recovers else -1]
        assert gone == {}
        assert list(present) == ["flaky"]


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_strict_mode_recognises_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(_plugins.STRICT_ENV_VAR, value)

    assert _plugins.strict_plugins_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe", None])
def test_strict_mode_rejects_other_values_and_unset(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(_plugins.STRICT_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(_plugins.STRICT_ENV_VAR, value)

    assert _plugins.strict_plugins_enabled() is False


@pytest.mark.usefixtures("unlatched", "ledgers", "strict")
def test_strict_mode_re_raises_records_the_failure_and_stops_loading(entry_points: _EntryPointRecorder) -> None:
    entry_points.entry_points.extend([_broken(), _StubEntryPoint("healthy")])

    with pytest.raises(RuntimeError, match="boom"):
        _plugins.discover_plugins()

    assert "broken" in _plugins.get_failed_plugins()
    assert _plugins._discovered is True
    assert entry_points.entry_points[1].loaded is False
