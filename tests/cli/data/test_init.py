"""Tests for the :mod:`drevalpy.cli.data` command group."""

from __future__ import annotations

import pytest

from drevalpy.cli.data import data_app
from drevalpy.cli.data.load import load_dataset
from drevalpy.cli.data.split import split_dataset
from drevalpy.cli.main import app
from tests.cli._helpers import HELP_ENV, make_runner, plain

runner = make_runner()


class TestGroup:
    """The group is help-first and exposes exactly two commands."""

    def test_app_name(self) -> None:
        assert data_app.info.name == "data"

    def test_registered_command_names(self) -> None:
        assert {command.name for command in data_app.registered_commands} == {"load", "split"}

    def test_commands_wrap_the_source_callbacks(self) -> None:
        callbacks = {command.name: command.callback for command in data_app.registered_commands}

        assert callbacks == {"load": load_dataset, "split": split_dataset}

    def test_bare_group_prints_help(self) -> None:
        result = runner.invoke(app, ["data"], env=HELP_ENV)

        assert "Usage" in plain(result.output)

    def test_bare_group_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["data"], env=HELP_ENV)

        assert result.exit_code != 0

    @pytest.mark.parametrize("command", ["load", "split"], ids=["load", "split"])
    def test_help_lists_each_command(self, command: str) -> None:
        result = runner.invoke(app, ["data", "--help"], env=HELP_ENV)

        assert command in plain(result.output)

    def test_unknown_subcommand_is_a_usage_error(self) -> None:
        result = runner.invoke(app, ["data", "not-a-command"], env=HELP_ENV)

        assert result.exit_code == 2
