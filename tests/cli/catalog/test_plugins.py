"""Tests for :mod:`drevalpy.cli.catalog.plugins`; the registry ledgers and entry points are patched."""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any

import pytest

from drevalpy.cli.catalog import plugins as plugins_module
from drevalpy.cli.main import app
from tests.cli._helpers import HELP_ENV, make_runner, plain

runner = make_runner()

TRACEBACK = 'Traceback (most recent call last):\n  File "x.py", line 1\n    boom\nRuntimeError: boom\n'
SKIPPED = "drevalpy.components.predictors.thing"


class StubEntryPoint:
    """Minimal stand-in for :class:`importlib.metadata.EntryPoint`."""

    def __init__(self, name: str, value: str) -> None:
        """Record the declared name and target."""
        self.name = name
        self.value = value


@pytest.fixture()
def declared(monkeypatch: pytest.MonkeyPatch) -> list[StubEntryPoint]:
    entries: list[StubEntryPoint] = []
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **_: entries)
    return entries


@pytest.fixture()
def ledgers(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, str]]:
    from drevalpy import registry

    state: dict[str, dict[str, str]] = {"loaded": {}, "failed": {}, "skipped": {}}
    monkeypatch.setattr(registry, "get_loaded_plugins", lambda: dict(state["loaded"]))
    monkeypatch.setattr(registry, "get_failed_plugins", lambda: dict(state["failed"]))
    monkeypatch.setattr(registry, "get_skipped_builtin_modules", lambda: dict(state["skipped"]))
    return state


def invoke(*argv: str) -> Any:
    return runner.invoke(app, ["list", "plugins", *argv], env=HELP_ENV)


def text(*argv: str) -> str:
    return plain(invoke(*argv).output)


def test_help_documents_the_options() -> None:
    helped = text("--help")
    assert "--traceback" in helped
    assert "--strict" in helped


def test_no_plugins_says_so_and_prints_no_table(declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
    assert invoke().exit_code == 0
    printed = text()
    assert "No packages declare a drevalpy.plugins entry point" in printed
    assert "installed into this interpreter" in printed
    assert "Status" not in printed


def test_a_loaded_plugin_is_listed_with_its_target(declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
    declared.append(StubEntryPoint("my_plugin", "my_plugin.register:setup"))
    ledgers["loaded"]["my_plugin"] = "my_plugin.register:setup"

    printed = text()

    assert "my_plugin" in printed
    assert plugins_module.STATUS_LOADED in printed
    assert "my_plugin.register:setup" in printed
    assert "failed to load" not in printed
    assert invoke("--strict").exit_code == 0


def test_a_failed_plugin_is_named_with_its_reason(declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
    declared.append(StubEntryPoint("broken_plugin", "broken_plugin:setup"))
    ledgers["failed"]["broken_plugin"] = TRACEBACK

    printed = text()

    assert plugins_module.STATUS_FAILED in printed
    assert "1 plugin(s) failed to load" in printed
    assert "RuntimeError: boom" in printed
    assert "Traceback (most recent call last)" not in printed
    assert "--traceback" in printed
    assert invoke().exit_code == 0
    assert invoke("--strict").exit_code == 1
    assert "RuntimeError: boom" in text("--strict")


def test_traceback_option_prints_the_stack_and_drops_the_advert(
    declared: list[StubEntryPoint], ledgers: dict[str, Any]
) -> None:
    ledgers["failed"]["broken_plugin"] = TRACEBACK

    printed = text("--traceback")

    assert "Traceback (most recent call last)" in printed
    assert "Re-run with --traceback" not in printed


def test_a_declared_plugin_in_neither_ledger_is_not_loaded_and_not_a_failure(
    declared: list[StubEntryPoint], ledgers: dict[str, Any]
) -> None:
    declared.append(StubEntryPoint("unseen", "unseen:setup"))

    assert plugins_module.STATUS_NOT_LOADED in text()
    assert "failed to load" not in text()
    assert invoke("--strict").exit_code == 0


@pytest.mark.parametrize("ledger", ["loaded", "failed"])
def test_a_ledger_entry_missing_from_metadata_is_still_listed(
    declared: list[StubEntryPoint], ledgers: dict[str, Any], ledger: str
) -> None:
    ledgers[ledger]["ghost"] = TRACEBACK

    assert "ghost" in text()


def test_rows_and_failures_are_sorted_by_name(declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
    for name in ("zeta", "alpha", "mu"):
        declared.append(StubEntryPoint(name, f"{name}:setup"))
        ledgers["loaded"][name] = f"{name}:setup"
    printed = text()
    assert printed.index("alpha") < printed.index("mu") < printed.index("zeta")

    ledgers["loaded"].clear()
    for name in ("zeta", "alpha"):
        ledgers["failed"][name] = f"ValueError: {name} exploded"
    printed = text()
    assert printed.index("alpha exploded") < printed.index("zeta exploded")


def test_skipped_builtins_are_reported_but_are_not_plugin_failures(
    declared: list[StubEntryPoint], ledgers: dict[str, Any]
) -> None:
    assert "skipped" not in text()

    ledgers["skipped"][SKIPPED] = TRACEBACK
    printed = text()

    assert SKIPPED in printed
    assert "1 built-in module(s) were skipped" in printed
    assert "RuntimeError: boom" in printed
    assert "failed to load" not in printed
    assert invoke("--strict").exit_code == 0


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [
        (TRACEBACK, "RuntimeError: boom"),
        ("ValueError: nope\n\n\n", "ValueError: nope"),
        ("ImportError: no module", "ImportError: no module"),
        ("", "unknown error"),
        ("   \n  \n", "unknown error"),
    ],
)
def test_failure_reason_keeps_the_last_line(recorded: str, expected: str) -> None:
    assert plugins_module.failure_reason(recorded) == expected


def test_a_registry_import_failure_exits_one_and_explains(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.import_module

    def fake(name: str, package: str | None = None) -> Any:
        if name == "drevalpy.registry":
            raise RuntimeError("plugin blew up during discovery")
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", fake)

    result = invoke()

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    printed = plain(result.output)
    assert "Importing drevalpy.registry failed while loading plugins." in printed
    assert "plugin blew up during discovery" in printed
