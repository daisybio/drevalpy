"""Shared fixtures-free helpers for the ``drevalpy.cli`` test modules.

Every CLI command body imports its heavy dependencies lazily, so the testing
lever throughout ``tests/cli`` is to monkeypatch the worker in its *source*
module (``drevalpy._run.run``, ``drevalpy.data.split``, ...) and assert on the
kwargs the command forwarded. The stubs below stand in for the objects those
workers return.

Note that typer >= 0.26 follows click 8.2, where ``mix_stderr`` is gone and
stderr is merged into ``result.output``; assertions on error text therefore use
``result.output`` rather than ``result.stderr``.
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

#: Rich renders help output with colour whenever a TTY-ish env is detected.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

#: A wide terminal keeps rich from wrapping option names mid-token.
HELP_ENV = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}


def make_runner() -> CliRunner:
    """Build a fresh ``CliRunner``.

    :returns: A runner with click's default exception catching enabled, so failures
        surface as ``result.exit_code`` plus ``result.exception``.
    """
    return CliRunner()


def plain(text: str) -> str:
    """Strip terminal escape codes so help-text assertions are colour-agnostic.

    :param text: Captured CLI output.
    :returns: ``text`` without ANSI escape sequences.
    """
    return _ANSI_ESCAPE.sub("", text)


def patch_worker(monkeypatch: pytest.MonkeyPatch, module_name: str, attribute: str, value: Any) -> None:
    """Replace ``module_name.attribute`` given the module object, not a dotted string.

    The worker modules are private (``drevalpy._run``, ``drevalpy._single``,
    ``drevalpy.experiment._randomization``, ...) precisely so that the public
    ``drevalpy.run`` / ``drevalpy.single`` names can be the re-exported
    *functions* without shadowing a module. Resolving the private module through
    :func:`importlib.import_module` patches the object that a command body's
    ``from drevalpy._run import run`` actually reads.

    :param monkeypatch: Fixture performing the (reverted) assignment.
    :param module_name: Dotted name of the module owning the worker.
    :param attribute: Name of the worker within that module.
    :param value: Replacement callable.
    """
    monkeypatch.setattr(importlib.import_module(module_name), attribute, value)


class FakeMuData:
    """Minimal ``mdata`` stand-in that records the paths it was asked to write."""

    def __init__(self) -> None:
        """Start with an empty write log."""
        self.written: list[str] = []

    def write(self, path: str) -> None:
        """Record ``path`` and create a placeholder file there.

        :param path: Destination the command chose for this dataset.
        """
        self.written.append(path)
        UPath(path).write_text("stub-h5mu")


class FakeDataset:
    """Stand-in for :class:`drevalpy.types.data.dataset.Dataset`.

    Only the attributes the CLI touches are implemented: ``name`` for the echoed
    summary, ``save`` (which writes through ``mdata``) for the write-out and
    ``randomization`` for the filename that ``experiments randomization`` derives.
    """

    def __init__(
        self,
        name: str = "StubDataset",
        randomization: tuple[str, str] | None = None,
    ) -> None:
        """Create a dataset stub.

        :param name: Value exposed as ``Dataset.name``.
        :param randomization: Value exposed as ``Dataset.randomization``.
        """
        self.name = name
        self.randomization = randomization
        self.mdata = FakeMuData()

    def save(self, path: str | UPath) -> None:
        """Write the placeholder file the way :meth:`Dataset.save` writes its MuData.

        :param path: Destination the command chose for this dataset.
        """
        self.mdata.write(str(path))


class Recorder:
    """Callable that records every call's positional and keyword arguments."""

    def __init__(self, return_value: Any = None) -> None:
        """Create a recorder.

        :param return_value: Value handed back to the caller on every invocation.
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


def write_synthetic_fold(runner: CliRunner, directory: UPath) -> tuple[UPath, UPath]:
    """Write a synthetic dataset and its first LPO fold through the CLI, as a user would.

    :param runner: Runner that invokes the ``drevalpy`` app.
    :param directory: Existing directory that receives ``synthetic.h5mu`` and ``folds/fold_0.npz``.
    :returns: The dataset path and the first fold's path.
    """
    from drevalpy.cli.main import app

    dataset = directory / "synthetic.h5mu"
    assert runner.invoke(app, ["data", "synthetic", "--output", str(dataset)]).exit_code == 0
    folds = directory / "folds"
    assert runner.invoke(app, ["data", "split", str(dataset), str(folds), "-n", "2"]).exit_code == 0
    return dataset, folds / "fold_0.npz"


class GroupSurface:
    """Tests every ``drevalpy`` command group shares; a subclass records which group it pins.

    Name the subclass ``TestGroup``. This base is not collected itself because its name does not start
    with ``Test``.
    """

    group: ClassVar[typer.Typer]
    path: ClassVar[str]
    callbacks: ClassVar[dict[str, Callable[..., Any]]]

    def test_app_name(self) -> None:
        assert self.group.info.name == self.path

    def test_registered_command_names(self) -> None:
        assert {command.name for command in self.group.registered_commands} == set(self.callbacks)

    def test_has_no_nested_groups(self) -> None:
        assert self.group.registered_groups == []

    def test_commands_wrap_the_source_callbacks(self) -> None:
        wrapped = {command.name: command.callback for command in self.group.registered_commands}

        assert wrapped == self.callbacks

    def test_bare_group_prints_help_and_exits_nonzero(self) -> None:
        from drevalpy.cli.main import app

        result = make_runner().invoke(app, [self.path], env=HELP_ENV)

        assert "Usage" in plain(result.output)
        assert result.exit_code != 0

    def test_help_lists_each_command(self) -> None:
        from drevalpy.cli.main import app

        result = make_runner().invoke(app, [self.path, "--help"], env=HELP_ENV)

        assert all(command in plain(result.output) for command in self.callbacks)

    def test_unknown_subcommand_is_a_usage_error(self) -> None:
        from drevalpy.cli.main import app

        result = make_runner().invoke(app, [self.path, "not-a-command"], env=HELP_ENV)

        assert result.exit_code == 2
