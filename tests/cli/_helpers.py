"""Shared fixtures-free helpers for the ``drevalpy.cli`` test modules.

Every CLI command body imports its heavy dependencies lazily, so the testing
lever throughout ``tests/cli`` is to monkeypatch the worker in its *source*
module (``drevalpy.run.run``, ``drevalpy.data.split``, ...) and assert on the
kwargs the command forwarded. The stubs below stand in for the objects those
workers return.

Note that typer >= 0.26 follows click 8.2, where ``mix_stderr`` is gone and
stderr is merged into ``result.output``; assertions on error text therefore use
``result.output`` rather than ``result.stderr``.
"""

from __future__ import annotations

import importlib
import re
from typing import Any

import pytest
from typer.testing import CliRunner
from upath import UPath

#: Rich renders help output with colour whenever a TTY-ish env is detected.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

#: A wide terminal keeps rich from wrapping option names mid-token.
HELP_ENV = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}


def make_runner() -> CliRunner:
    """Build a fresh ``CliRunner``.

    Returns:
        A runner with click's default exception catching enabled, so failures
        surface as ``result.exit_code`` plus ``result.exception``.
    """
    return CliRunner()


def plain(text: str) -> str:
    """Strip terminal escape codes so help-text assertions are colour-agnostic.

    Args:
        text: Captured CLI output.

    Returns:
        ``text`` without ANSI escape sequences.
    """
    return _ANSI_ESCAPE.sub("", text)


def patch_worker(monkeypatch: pytest.MonkeyPatch, module_name: str, attribute: str, value: Any) -> None:
    """Replace ``module_name.attribute`` given the module object, not a dotted string.

    ``drevalpy/__init__.py`` re-exports ``run``, ``single``, ``split`` and
    ``randomization`` as *functions*, which shadows the same-named submodules on
    the package object. ``monkeypatch.setattr("drevalpy.run.run", ...)``
    therefore walks into the function and fails; resolving the module through
    :func:`importlib.import_module` sidesteps the shadowing, and matches what
    ``from drevalpy.run import run`` inside a command body actually reads.

    Args:
        monkeypatch: Fixture performing the (reverted) assignment.
        module_name: Dotted name of the module owning the worker.
        attribute: Name of the worker within that module.
        value: Replacement callable.
    """
    monkeypatch.setattr(importlib.import_module(module_name), attribute, value)


class FakeMuData:
    """Minimal ``mdata`` stand-in that records the paths it was asked to write."""

    def __init__(self) -> None:
        """Start with an empty write log."""
        self.written: list[str] = []

    def write(self, path: str) -> None:
        """Record ``path`` and create a placeholder file there.

        Args:
            path: Destination the command chose for this dataset.
        """
        self.written.append(path)
        UPath(path).write_text("stub-h5mu")


class FakeDataset:
    """Stand-in for :class:`drevalpy.types.data.dataset.Dataset`.

    Only the attributes the CLI touches are implemented: ``name`` for the echoed
    summary, ``mdata`` for the write-out and ``randomization`` for the filename
    that ``experiments randomization`` derives.
    """

    def __init__(
        self,
        name: str = "StubDataset",
        randomization: tuple[str, str] | None = None,
    ) -> None:
        """Create a dataset stub.

        Args:
            name: Value exposed as ``Dataset.name``.
            randomization: Value exposed as ``Dataset.randomization``.
        """
        self.name = name
        self.randomization = randomization
        self.mdata = FakeMuData()


class Recorder:
    """Callable that records every call's positional and keyword arguments."""

    def __init__(self, return_value: Any = None) -> None:
        """Create a recorder.

        Args:
            return_value: Value handed back to the caller on every invocation.
        """
        self.return_value = return_value
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record the call and return the configured value."""
        self.calls.append((args, kwargs))
        return self.return_value

    @property
    def call_count(self) -> int:
        """Number of times the recorder was invoked."""
        return len(self.calls)

    @property
    def args(self) -> tuple[Any, ...]:
        """Positional arguments of the single recorded call."""
        assert self.call_count == 1, f"expected exactly one call, got {self.call_count}"
        return self.calls[0][0]

    @property
    def kwargs(self) -> dict[str, Any]:
        """Keyword arguments of the single recorded call."""
        assert self.call_count == 1, f"expected exactly one call, got {self.call_count}"
        return self.calls[0][1]
