"""Tests for the :mod:`drevalpy.cli.catalog` command group surface."""

from __future__ import annotations

import pytest

from drevalpy.cli import catalog
from drevalpy.cli.catalog import list_app
from drevalpy.cli.catalog import plugins as plugins_module
from drevalpy.cli.catalog import registries as registries_module
from drevalpy.cli.main import app
from tests.cli._helpers import HELP_ENV, GroupSurface, make_runner, plain

runner = make_runner()

CALLBACKS = {
    "predictors": registries_module.list_predictors,
    "sample-featurizers": registries_module.list_sample_featurizers,
    "drug-featurizers": registries_module.list_drug_featurizers,
    "splitters": registries_module.list_splitters,
    "visualizations": registries_module.list_visualizations,
    "zoo-presets": registries_module.list_zoo_presets,
    "datasets": registries_module.list_datasets,
    "plugins": plugins_module.list_plugins,
}


class TestGroup(GroupSurface):
    group = list_app
    path = "list"
    callbacks = CALLBACKS

    @pytest.mark.parametrize("command", CALLBACKS, ids=CALLBACKS)
    def test_dash_h_reaches_each_command(self, command: str) -> None:
        result = runner.invoke(app, ["list", command, "-h"], env=HELP_ENV)

        assert result.exit_code == 0
        assert "Usage" in plain(result.output)

    def test_all_lists_only_the_app(self) -> None:
        assert catalog.__all__ == ["list_app"]
