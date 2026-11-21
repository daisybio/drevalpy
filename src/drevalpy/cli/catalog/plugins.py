"""``drevalpy list plugins`` command: which installed plugins loaded, and why not.

A plugin that raises on import silently removes every component it would have
registered, which otherwise resurfaces much later as an "unknown predictor"
error. :func:`drevalpy.registry.get_failed_plugins` records those failures; this
command is where they become visible, so it doubles as the smoke test for a
plugin repository's CI.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import ModuleType
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from rich.text import Text


class PluginStatus(StrEnum):
    """Whether a declared plugin imported, and the style its row is printed in."""

    LOADED = "loaded"
    FAILED = "failed"
    NOT_LOADED = "not loaded"

    @property
    def style(self) -> str:
        """The ``rich`` style of this status in the plugin table."""
        return {"loaded": "green", "failed": "bold red", "not loaded": "yellow"}[self.value]


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

    :returns: The imported :mod:`drevalpy.registry` module.
    :raises typer.Exit: With code 1 when the import failed.
    """
    import importlib

    from ._render import console

    try:
        return importlib.import_module("drevalpy.registry")
    except Exception as error:  # strict mode surfaces the plugin's own exception
        import traceback

        from rich.text import Text

        out = console()
        out.print(Text("Importing drevalpy.registry failed while loading plugins.", style="bold red"))
        out.print(Text(traceback.format_exc().rstrip()))
        raise typer.Exit(1) from error


def failure_reason(traceback_text: str) -> str:
    """Reduce a formatted traceback to its actionable last line.

    :param traceback_text: Traceback as recorded by
        :func:`drevalpy.registry.get_failed_plugins`.
    :returns: The exception line (type and message), or a placeholder when the
        traceback is empty.
    """
    lines = [line.strip() for line in traceback_text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "unknown error"


def _status_rows(
    declared: Mapping[str, str],
    loaded: Mapping[str, str],
    failed: Mapping[str, str],
) -> list[list[str | Text]]:
    """Build the plugin status table rows.

    :param declared: Entry points found in installed distribution metadata.
    :param loaded: Plugins that imported cleanly, mapped to their entry-point value.
    :param failed: Plugins that raised while importing, mapped to their traceback.
    :returns: One ``[name, status, entry point]`` row per plugin, sorted by name.
    """
    from rich.text import Text

    rows: list[list[str | Text]] = []
    for name in sorted(set(declared) | set(loaded) | set(failed)):
        status = (
            PluginStatus.FAILED
            if name in failed
            else PluginStatus.LOADED
            if name in loaded
            else PluginStatus.NOT_LOADED
        )
        target = declared.get(name) or loaded.get(name) or ""
        rows.append([name, Text(status, style=status.style), target])
    return rows


def _report(items: Mapping[str, str], heading: str, style: str, *, show_traceback: bool | None = None) -> None:
    """Print a heading and the one-line reason for every recorded failure.

    :param items: Mapping of name to the recorded traceback; nothing is printed when empty.
    :param heading: Text after the count in the heading line.
    :param style: ``rich`` style of the heading and the names.
    :param show_traceback: Print each whole traceback rather than its last line only;
        ``None`` for a report that offers no ``--traceback`` option.
    """
    if not items:
        return
    from rich.text import Text

    from ._render import console

    out = console()
    out.print()
    out.print(Text(f"{len(items)} {heading}:", style=style))
    for name, recorded in sorted(items.items()):
        line = Text("  ")
        line.append(name, style=style)
        line.append(": ")
        line.append(failure_reason(recorded))
        out.print(line)
        if show_traceback:
            out.print(Text(recorded.rstrip(), style="dim"))
    if show_traceback is False:
        out.print(Text("Re-run with --traceback for the full stack traces.", style="dim"))


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
    from drevalpy.registry._plugins import declared_entry_points

    from ._render import render_rows

    registry = _registry_module()
    failed = registry.get_failed_plugins()

    render_rows(
        _status_rows(
            {name: ep.value for name, ep in declared_entry_points().items()}, registry.get_loaded_plugins(), failed
        ),
        columns=["Plugin", "Status", "Entry point"],
        title="Plugins",
        empty_hint=_NO_PLUGINS_HINT,
    )
    _report(failed, "plugin(s) failed to load", "bold red", show_traceback=show_traceback)
    # Builtins are not plugins, but a missing built-in component looks identical from the outside.
    _report(registry.get_skipped_builtin_modules(), "built-in module(s) were skipped", "yellow")

    if strict and failed:
        raise typer.Exit(1)
