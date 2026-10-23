"""Tests for the :mod:`drevalpy.cli.experiments` command group."""

from __future__ import annotations

from drevalpy.cli.experiments import experiments_app
from drevalpy.cli.experiments.randomization import randomization_cmd
from drevalpy.cli.experiments.robustness import robustness_cmd
from tests.cli._helpers import GroupSurface


class TestGroup(GroupSurface):
    group = experiments_app
    path = "experiments"
    callbacks = {"robustness": robustness_cmd, "randomization": randomization_cmd}
