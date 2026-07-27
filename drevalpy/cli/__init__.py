"""Typer-based CLI for drevalpy."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["app", "cli_main"]

if TYPE_CHECKING:
    import typer

    app: typer.Typer


def __getattr__(name: str):
    if name in {"app", "cli_main"}:
        from drevalpy.cli.main import app as _app
        from drevalpy.cli.main import cli_main as _cli_main

        exports = {"app": _app, "cli_main": _cli_main}
        value = exports[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
