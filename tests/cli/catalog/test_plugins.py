"""Tests for :mod:`drevalpy.cli.catalog.plugins`, the ``drevalpy list plugins`` command.

The command reads three process-global ledgers -
:func:`~drevalpy.registry.get_loaded_plugins`,
:func:`~drevalpy.registry.get_failed_plugins` and
:func:`~drevalpy.registry.get_skipped_builtin_modules` - plus the entry points
declared in installed distribution metadata. A real environment has no broken
plugin in it, so the interesting states are produced by patching those four
boundaries; that is also why the ledgers are patched rather than mutated, so the
rest of the suite sees them unchanged.
"""

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


class StubEntryPoint:
    """Minimal stand-in for :class:`importlib.metadata.EntryPoint`."""

    def __init__(self, name: str, value: str) -> None:
        """Record the declared name and target.

        Args:
            name: Entry-point name, as it appears in the distribution metadata.
            value: Dotted object reference the plugin declared.
        """
        self.name = name
        self.value = value


@pytest.fixture()
def declared(monkeypatch: pytest.MonkeyPatch) -> list[StubEntryPoint]:
    """Replace entry-point discovery with a list the test can fill.

    Args:
        monkeypatch: Fixture used to patch the ``importlib.metadata`` boundary.

    Returns:
        The mutable list of entry points the command will see.
    """
    entries: list[StubEntryPoint] = []
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **_: entries)
    return entries


@pytest.fixture()
def ledgers(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, str]]:
    """Give the command empty loaded/failed/skipped ledgers it can be handed.

    Args:
        monkeypatch: Fixture used to replace the registry accessors.

    Returns:
        Mapping with ``loaded``, ``failed`` and ``skipped`` dicts, each mutable
        and read by the command on every invocation.
    """
    from drevalpy import registry

    state = {"loaded": {}, "failed": {}, "skipped": {}}
    monkeypatch.setattr(registry, "get_loaded_plugins", lambda: dict(state["loaded"]))
    monkeypatch.setattr(registry, "get_failed_plugins", lambda: dict(state["failed"]))
    monkeypatch.setattr(registry, "get_skipped_builtin_modules", lambda: dict(state["skipped"]))
    return state


def invoke(*argv: str) -> Any:
    """Run ``drevalpy list plugins`` and return the click result.

    Args:
        *argv: Extra arguments appended after ``plugins``.

    Returns:
        The ``click.testing.Result`` of the invocation.
    """
    return runner.invoke(app, ["list", "plugins", *argv], env=HELP_ENV)


def text(*argv: str) -> str:
    """Return the plain-text output of ``drevalpy list plugins``.

    Args:
        *argv: Extra arguments appended after ``plugins``.

    Returns:
        Output with escape codes stripped.
    """
    return plain(invoke(*argv).output)


class TestOptions:
    """The two options are documented in the command's own help.

    That the command is wired into the group at all is pinned in ``test_init.py``.
    """

    def test_help_documents_the_traceback_option(self) -> None:
        assert "--traceback" in text("--help")

    def test_help_documents_the_strict_option(self) -> None:
        assert "--strict" in text("--help")


class TestNoPlugins:
    """An environment without plugins says so, and says how to check."""

    def test_exits_cleanly(self, declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
        assert invoke().exit_code == 0

    def test_explains_that_nothing_declares_the_entry_point(
        self, declared: list[StubEntryPoint], ledgers: dict[str, Any]
    ) -> None:
        assert "No packages declare a drevalpy.plugins entry point" in text()

    def test_names_the_likely_cause(self, declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
        assert "installed into this interpreter" in text()

    def test_prints_no_table(self, declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
        assert "Status" not in text()


class TestLoadedPlugin:
    """A healthy plugin is listed as loaded, with the object it points at."""

    @pytest.fixture(autouse=True)
    def healthy(self, declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
        """Declare one plugin and mark it loaded."""
        declared.append(StubEntryPoint("my_plugin", "my_plugin.register:setup"))
        ledgers["loaded"]["my_plugin"] = "my_plugin.register:setup"

    def test_exits_cleanly(self) -> None:
        assert invoke().exit_code == 0

    def test_lists_the_plugin_name(self) -> None:
        assert "my_plugin" in text()

    def test_reports_the_loaded_status(self) -> None:
        assert plugins_module.STATUS_LOADED in text()

    def test_shows_the_entry_point_target(self) -> None:
        assert "my_plugin.register:setup" in text()

    def test_reports_no_failures(self) -> None:
        assert "failed to load" not in text()

    def test_strict_mode_still_succeeds(self) -> None:
        assert invoke("--strict").exit_code == 0


class TestFailedPlugin:
    """A plugin that raised is named together with the reason it raised."""

    @pytest.fixture(autouse=True)
    def broken(self, declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
        """Declare one plugin and record an import failure for it."""
        declared.append(StubEntryPoint("broken_plugin", "broken_plugin:setup"))
        ledgers["failed"]["broken_plugin"] = TRACEBACK

    def test_reports_the_failed_status(self) -> None:
        assert plugins_module.STATUS_FAILED in text()

    def test_names_the_plugin_in_the_failure_report(self) -> None:
        assert "broken_plugin" in text()

    def test_reports_how_many_failed(self) -> None:
        assert "1 plugin(s) failed to load" in text()

    def test_surfaces_the_exception_line(self) -> None:
        """The last traceback line is the actionable part; it must be visible."""
        assert "RuntimeError: boom" in text()

    def test_omits_the_stack_frames_by_default(self) -> None:
        assert "Traceback (most recent call last)" not in text()

    def test_advertises_the_traceback_option(self) -> None:
        assert "--traceback" in text()

    def test_traceback_option_prints_the_stack(self) -> None:
        assert "Traceback (most recent call last)" in text("--traceback")

    def test_traceback_option_drops_the_advert(self) -> None:
        assert "Re-run with --traceback" not in text("--traceback")

    def test_default_exit_code_is_zero(self) -> None:
        """Reporting a broken plugin is the command working, not failing."""
        assert invoke().exit_code == 0

    def test_strict_option_makes_a_failure_fatal(self) -> None:
        assert invoke("--strict").exit_code == 1

    def test_strict_option_still_prints_the_report(self) -> None:
        assert "RuntimeError: boom" in text("--strict")


class TestNotLoadedPlugin:
    """A declared entry point in neither ledger is reported as not loaded.

    This is what a plugin looks like when discovery has not run in the current
    process, so it must not be silently rendered as healthy.
    """

    @pytest.fixture(autouse=True)
    def undiscovered(self, declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
        """Declare a plugin without recording either outcome for it."""
        declared.append(StubEntryPoint("unseen", "unseen:setup"))

    def test_reports_the_not_loaded_status(self) -> None:
        assert plugins_module.STATUS_NOT_LOADED in text()

    def test_is_not_counted_as_a_failure(self) -> None:
        assert "failed to load" not in text()

    def test_strict_mode_ignores_it(self) -> None:
        assert invoke("--strict").exit_code == 0


class TestUndeclaredButLoaded:
    """A ledger entry with no matching metadata is still listed.

    Plugin metadata and the ledgers are read from different places, so they can
    disagree; dropping the difference would hide the more surprising half.
    """

    def test_a_loaded_plugin_missing_from_metadata_is_listed(
        self, declared: list[StubEntryPoint], ledgers: dict[str, Any]
    ) -> None:
        ledgers["loaded"]["ghost"] = "ghost:setup"

        assert "ghost" in text()

    def test_a_failed_plugin_missing_from_metadata_is_listed(
        self, declared: list[StubEntryPoint], ledgers: dict[str, Any]
    ) -> None:
        ledgers["failed"]["ghost"] = TRACEBACK

        assert "ghost" in text()


class TestOrdering:
    """Plugins are listed in a stable, name-sorted order."""

    def test_rows_are_sorted_by_name(self, declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
        for name in ("zeta", "alpha", "mu"):
            declared.append(StubEntryPoint(name, f"{name}:setup"))
            ledgers["loaded"][name] = f"{name}:setup"
        printed = text()

        assert printed.index("alpha") < printed.index("mu") < printed.index("zeta")

    def test_failures_are_sorted_by_name(self, declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
        for name in ("zeta", "alpha"):
            declared.append(StubEntryPoint(name, f"{name}:setup"))
            ledgers["failed"][name] = f"ValueError: {name} exploded"
        printed = text()

        assert printed.index("alpha exploded") < printed.index("zeta exploded")


class TestSkippedBuiltins:
    """A skipped built-in module looks like a missing component too."""

    def test_is_reported(self, declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
        ledgers["skipped"]["drevalpy.components.predictors.thing"] = TRACEBACK

        assert "drevalpy.components.predictors.thing" in text()

    def test_reports_how_many_were_skipped(self, declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
        ledgers["skipped"]["drevalpy.components.predictors.thing"] = TRACEBACK

        assert "1 built-in module(s) were skipped" in text()

    def test_surfaces_the_reason(self, declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
        ledgers["skipped"]["drevalpy.components.predictors.thing"] = TRACEBACK

        assert "RuntimeError: boom" in text()

    def test_is_not_counted_as_a_plugin_failure(self, declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
        ledgers["skipped"]["drevalpy.components.predictors.thing"] = TRACEBACK

        assert "failed to load" not in text()

    def test_does_not_make_strict_mode_fail(self, declared: list[StubEntryPoint], ledgers: dict[str, Any]) -> None:
        """``--strict`` is about the caller's plugins, not drevalpy's own builtins."""
        ledgers["skipped"]["drevalpy.components.predictors.thing"] = TRACEBACK

        assert invoke("--strict").exit_code == 0

    def test_nothing_is_printed_when_none_were_skipped(
        self, declared: list[StubEntryPoint], ledgers: dict[str, Any]
    ) -> None:
        assert "skipped" not in text()


class TestFailureReason:
    """``failure_reason`` reduces a traceback to its last, useful line."""

    def test_returns_the_exception_line(self) -> None:
        assert plugins_module.failure_reason(TRACEBACK) == "RuntimeError: boom"

    def test_ignores_trailing_blank_lines(self) -> None:
        assert plugins_module.failure_reason("ValueError: nope\n\n\n") == "ValueError: nope"

    def test_handles_a_single_line(self) -> None:
        assert plugins_module.failure_reason("ImportError: no module") == "ImportError: no module"

    @pytest.mark.parametrize("recorded", ["", "\n", "   \n  \n"], ids=["empty", "newline", "whitespace"])
    def test_empty_input_yields_a_placeholder(self, recorded: str) -> None:
        assert plugins_module.failure_reason(recorded) == "unknown error"


class TestRegistryImportFailure:
    """With strict mode on, importing the registry is itself what fails."""

    @pytest.fixture()
    def broken_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make importing :mod:`drevalpy.registry` raise the way a plugin would.

        Only that one name is diverted; everything else still imports normally,
        which matters because the failure path itself imports rich.
        """
        real = importlib.import_module

        def fake(name: str, package: str | None = None) -> Any:
            if name == "drevalpy.registry":
                raise RuntimeError("plugin blew up during discovery")
            return real(name, package)

        monkeypatch.setattr(importlib, "import_module", fake)

    def test_exit_code_is_one(self, broken_import: None) -> None:
        assert invoke().exit_code == 1

    def test_explains_what_failed(self, broken_import: None) -> None:
        assert "Importing drevalpy.registry failed while loading plugins." in text()

    def test_prints_the_underlying_error(self, broken_import: None) -> None:
        assert "plugin blew up during discovery" in text()

    def test_does_not_escape_as_an_unhandled_exception(self, broken_import: None) -> None:
        assert isinstance(invoke().exception, SystemExit)
