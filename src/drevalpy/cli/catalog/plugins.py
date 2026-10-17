"""``drevalpy list plugins`` command: which installed plugins loaded, and why not.

A plugin that raises on import silently removes every component it would have
registered, which otherwise resurfaces much later as an "unknown predictor"
error. :func:`drevalpy.registry.get_failed_plugins` records those failures; this
command is where they become visible, so it doubles as the smoke test for a
plugin repository's CI.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from rich.text import Text

STATUS_LOADED = "loaded"
STATUS_FAILED = "failed"
STATUS_NOT_LOADED = "not loaded"

_STATUS_STYLES = {
    STATUS_LOADED: "green",
    STATUS_FAILED: "bold red",
    STATUS_NOT_LOADED: "yellow",
}

_NO_PLUGINS_HINT = (
    "No packages declare a drevalpy.plugins entry point in this environment. "
    "If you expected one, check that it is installed into this interpreter."
)


def _registry_module() -> ModuleType:
    """Import :mod:`drevalpy.registry`, reporting a strict-mode failure cleanly.

    With ``DREVALPY_STRICT_PLUGINS`` set, plugin discovery re-raises, and that
    happens on import of the registry package. Diagnosing a broken plugin is
    exactly what this command is for, so the traceback is printed as output
    instead of escaping as an unhandled crash.

    Returns:
        The imported :mod:`drevalpy.registry` module.

    Raises:
        typer.Exit: With code 1 when the import failed.
    """
    import importlib

    from ._render import console

    try:
        return importlib.import_module("drevalpy.registry")
    except Exception as error:  # noqa: BLE001 - strict mode surfaces the plugin's own exception
        import traceback

        from rich.text import Text

        out = console()
        out.print(Text("Importing drevalpy.registry failed while loading plugins.", style="bold red"))
        out.print(Text(traceback.format_exc().rstrip()))
        raise typer.Exit(1) from error


def _declared_entry_points() -> dict[str, str]:
    """Return the ``drevalpy.plugins`` entry points declared in this environment.

    Returns:
        Mapping of entry-point name to the dotted object it points at. Includes
        entry points that failed to load, which is why it is read from
        ``importlib.metadata`` rather than from the loaded-plugins registry.
    """
    import importlib.metadata

    # The group name lives with the loader that consumes it; duplicating the
    # string here would let the two drift apart silently.
    from drevalpy.registry._plugins import ENTRY_POINT_GROUP

    return {ep.name: ep.value for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)}


def failure_reason(traceback_text: str) -> str:
    """Reduce a formatted traceback to its actionable last line.

    Args:
        traceback_text: Traceback as recorded by
            :func:`drevalpy.registry.get_failed_plugins`.

    Returns:
        The exception line (type and message), or a placeholder when the
        traceback is empty.
    """
    lines = [line.strip() for line in traceback_text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "unknown error"


def _status(name: str, loaded: Mapping[str, str], failed: Mapping[str, str]) -> str:
    """Classify one entry point.

    Args:
        name: Entry-point name.
        loaded: Plugins that imported cleanly.
        failed: Plugins that raised while importing.

    Returns:
        One of :data:`STATUS_FAILED`, :data:`STATUS_LOADED` or
        :data:`STATUS_NOT_LOADED`.
    """
    if name in failed:
        return STATUS_FAILED
    if name in loaded:
        return STATUS_LOADED
    return STATUS_NOT_LOADED


def _status_rows(
    declared: Mapping[str, str],
    loaded: Mapping[str, str],
    failed: Mapping[str, str],
) -> list[list[str | Text]]:
    """Build the plugin status table rows.

    Args:
        declared: Entry points found in installed distribution metadata.
        loaded: Plugins that imported cleanly, mapped to their entry-point value.
        failed: Plugins that raised while importing, mapped to their traceback.

    Returns:
        One ``[name, status, entry point]`` row per plugin, sorted by name.
    """
    from rich.text import Text

    rows: list[list[str | Text]] = []
    for name in sorted(set(declared) | set(loaded) | set(failed)):
        status = _status(name, loaded, failed)
        target = declared.get(name) or loaded.get(name) or ""
        rows.append([name, Text(status, style=_STATUS_STYLES[status]), target])
    return rows


def _report_failures(failed: Mapping[str, str], *, show_traceback: bool) -> None:
    """Print the reason each failed plugin failed.

    Args:
        failed: Mapping of plugin name to the recorded traceback.
        show_traceback: Print the whole traceback rather than its last line.
    """
    if not failed:
        return
    from rich.text import Text

    from ._render import console

    out = console()
    out.print()
    out.print(Text(f"{len(failed)} plugin(s) failed to load:", style="bold red"))
    for name, recorded in sorted(failed.items()):
        line = Text("  ")
        line.append(name, style="bold red")
        line.append(": ")
        line.append(failure_reason(recorded))
        out.print(line)
        if show_traceback:
            out.print(Text(recorded.rstrip(), style="dim"))
    if not show_traceback:
        out.print(Text("Re-run with --traceback for the full stack traces.", style="dim"))


def _report_skipped_builtins(skipped: Mapping[str, str]) -> None:
    """Print built-in modules that were skipped during registration.

    Builtins are not plugins, but a missing built-in component looks identical
    from the outside, so it belongs in the same answer.

    Args:
        skipped: Mapping of module name to the recorded traceback.
    """
    if not skipped:
        return
    from rich.text import Text

    from ._render import console

    out = console()
    out.print()
    out.print(Text(f"{len(skipped)} built-in module(s) were skipped:", style="yellow"))
    for module_name, recorded in sorted(skipped.items()):
        line = Text("  ")
        line.append(module_name, style="yellow")
        line.append(": ")
        line.append(failure_reason(recorded))
        out.print(line)


def list_plugins(
    show_traceback: Annotated[
        bool,
        typer.Option("--traceback", "-t", help="Print the full traceback of every failed plugin."),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit with code 1 when any declared plugin failed to load."),
    ] = False,
) -> None:
    """Show installed drevalpy plugins and whether they loaded.

    Every package declaring a ``drevalpy.plugins`` entry point is listed with its
    load status. Failures are reported with the exception that caused them; pass
    ``--strict`` to turn any failure into a non-zero exit status for CI.
    """
    from ._render import render_rows

    registry = _registry_module()
    failed = registry.get_failed_plugins()

    render_rows(
        _status_rows(_declared_entry_points(), registry.get_loaded_plugins(), failed),
        columns=["Plugin", "Status", "Entry point"],
        title="Plugins",
        empty_hint=_NO_PLUGINS_HINT,
    )
    _report_failures(failed, show_traceback=show_traceback)
    _report_skipped_builtins(registry.get_skipped_builtin_modules())

    if strict and failed:
        raise typer.Exit(1)
