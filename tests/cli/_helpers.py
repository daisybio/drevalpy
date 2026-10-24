"""Helpers for the CLI tests: monkeypatch a worker in its source module and assert on the forwarded kwargs.

Typer >= 0.26 merges stderr into ``result.output``, so error text is asserted on ``result.output``.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from typing import Any, ClassVar

import pytest
import typer
from typer.testing import CliRunner
from upath import UPath

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

HELP_ENV = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}


def make_runner() -> CliRunner:
    """Build a fresh ``CliRunner``."""
    return CliRunner()


def plain(text: str) -> str:
    """Strip terminal escape codes from captured CLI output."""
    return _ANSI_ESCAPE.sub("", text)


def patch_worker(monkeypatch: pytest.MonkeyPatch, module_name: str, attribute: str, value: Any) -> None:
    """Replace ``attribute`` on the private worker module ``module_name`` (resolved by import, not dotted string)."""
    monkeypatch.setattr(importlib.import_module(module_name), attribute, value)


class FakeMuData:
    """Minimal ``mdata`` stand-in that records the paths it was asked to write."""

    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, path: str) -> None:
        self.written.append(path)
        UPath(path).write_text("stub-h5mu")


class FakeDataset:
    """Stand-in for ``Dataset`` with only the attributes the CLI touches."""

    def __init__(
        self,
        name: str = "StubDataset",
        randomization: tuple[str, str] | None = None,
    ) -> None:
        self.name = name
        self.randomization = randomization
        self.mdata = FakeMuData()

    def save(self, path: str | UPath) -> None:
        self.mdata.write(str(path))


class Recorder:
    """Callable that records every call's positional and keyword arguments."""

    def __init__(self, return_value: Any = None) -> None:
        self.return_value = return_value
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
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


def write_synthetic_fold(runner: CliRunner, directory: UPath) -> tuple[UPath, UPath]:
    """Write a synthetic dataset and its first fold through the CLI; return both paths."""
    from drevalpy.cli.main import app

    dataset = directory / "synthetic.h5mu"
    assert runner.invoke(app, ["data", "synthetic", "--output", str(dataset)]).exit_code == 0
    folds = directory / "folds"
    assert runner.invoke(app, ["data", "split", str(dataset), str(folds), "-n", "2"]).exit_code == 0
    return dataset, folds / "fold_0.npz"


class GroupSurface:
    """Tests every ``drevalpy`` command group shares; name the subclass ``TestGroup`` and record the group."""

    group: ClassVar[typer.Typer]
    path: ClassVar[str]
    callbacks: ClassVar[dict[str, Callable[..., Any]]]

    def test_registered_command_names(self) -> None:
        assert {command.name for command in self.group.registered_commands} == set(self.callbacks)

    def test_bare_group_prints_help_and_exits_nonzero(self) -> None:
        from drevalpy.cli.main import app

        result = make_runner().invoke(app, [self.path], env=HELP_ENV)

        assert "Usage" in plain(result.output)
        assert result.exit_code != 0

    def test_help_lists_each_command_and_unknown_subcommand_is_a_usage_error(self) -> None:
        from drevalpy.cli.main import app

        helped = make_runner().invoke(app, [self.path, "--help"], env=HELP_ENV)
        unknown = make_runner().invoke(app, [self.path, "not-a-command"], env=HELP_ENV)

        assert all(command in plain(helped.output) for command in self.callbacks)
        assert unknown.exit_code == 2
