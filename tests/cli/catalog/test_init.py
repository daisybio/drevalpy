"""Tests for the :mod:`drevalpy.cli.catalog` command group surface."""

from __future__ import annotations

import pytest

from drevalpy.cli.catalog import list_app
from drevalpy.cli.catalog import plugins as plugins_module
from drevalpy.cli.catalog import registries as registries_module
from drevalpy.cli.main import app
from tests.cli._helpers import HELP_ENV, make_runner, plain

runner = make_runner()

#: The command surface documented in the plan, one per registry plus plugins.
COMMAND_NAMES = [
    "predictors",
    "cell-line-featurizers",
    "drug-featurizers",
    "splitters",
    "visualizations",
    "plugins",
]

EXPECTED_CALLBACKS = {
    "predictors": registries_module.list_predictors,
    "cell-line-featurizers": registries_module.list_cell_line_featurizers,
    "drug-featurizers": registries_module.list_drug_featurizers,
    "splitters": registries_module.list_splitters,
    "visualizations": registries_module.list_visualizations,
    "plugins": plugins_module.list_plugins,
}


class TestGroup:
    """The group is help-first and exposes exactly the six commands."""

    def test_app_name(self) -> None:
        assert list_app.info.name == "list"

    def test_registered_command_names(self) -> None:
        assert {command.name for command in list_app.registered_commands} == set(COMMAND_NAMES)

    def test_no_nested_groups(self) -> None:
        assert list_app.registered_groups == []

    def test_commands_wrap_the_source_callbacks(self) -> None:
        callbacks = {command.name: command.callback for command in list_app.registered_commands}

        assert callbacks == EXPECTED_CALLBACKS

    def test_bare_group_prints_help(self) -> None:
        result = runner.invoke(app, ["list"], env=HELP_ENV)

        assert "Usage" in plain(result.output)

    def test_bare_group_exits_nonzero(self) -> None:
        result = runner.invoke(app, ["list"], env=HELP_ENV)

        assert result.exit_code != 0

    @pytest.mark.parametrize("command", COMMAND_NAMES, ids=COMMAND_NAMES)
    def test_help_lists_each_command(self, command: str) -> None:
        result = runner.invoke(app, ["list", "--help"], env=HELP_ENV)

        assert command in plain(result.output)

    @pytest.mark.parametrize("command", COMMAND_NAMES, ids=COMMAND_NAMES)
    def test_dash_h_reaches_each_command(self, command: str) -> None:
        result = runner.invoke(app, ["list", command, "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert "Usage" in plain(result.output)

    def test_unknown_subcommand_is_a_usage_error(self) -> None:
        result = runner.invoke(app, ["list", "not-a-command"], env=HELP_ENV)

        assert result.exit_code == 2

    def test_all_lists_only_the_app(self) -> None:
        from drevalpy.cli import catalog

        assert catalog.__all__ == ["list_app"]
