"""Tests for the :mod:`drevalpy.cli.experiments` command group."""

from __future__ import annotations

import pytest

from drevalpy.cli.experiments import experiments_app
from drevalpy.cli.experiments.randomization import randomization_cmd
from drevalpy.cli.experiments.robustness import robustness_cmd
from drevalpy.cli.main import app
from tests.cli._helpers import HELP_ENV, make_runner, plain

runner = make_runner()


class TestGroup:
    """The group is help-first and exposes exactly two commands."""

    def test_app_name(self) -> None:
        assert experiments_app.info.name == "experiments"

    def test_registered_command_names(self) -> None:
        registered = {command.name for command in experiments_app.registered_commands}

        assert registered == {"robustness", "randomization"}

    def test_commands_wrap_the_source_callbacks(self) -> None:
        callbacks = {command.name: command.callback for command in experiments_app.registered_commands}

        assert callbacks == {"robustness": robustness_cmd, "randomization": randomization_cmd}

    def test_bare_group_prints_help(self) -> None:
        result = runner.invoke(app, ["experiments"], env=HELP_ENV)

        assert "Usage" in plain(result.output)

    def test_bare_group_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["experiments"], env=HELP_ENV)

        assert result.exit_code != 0

    @pytest.mark.parametrize("command", ["robustness", "randomization"], ids=["robustness", "randomization"])
    def test_help_lists_each_command(self, command: str) -> None:
        result = runner.invoke(app, ["experiments", "--help"], env=HELP_ENV)

        assert command in plain(result.output)

    def test_unknown_subcommand_is_a_usage_error(self) -> None:
        result = runner.invoke(app, ["experiments", "not-a-command"], env=HELP_ENV)

        assert result.exit_code == 2
