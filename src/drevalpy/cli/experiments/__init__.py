"""``drevalpy experiments`` command group."""

from __future__ import annotations

import typer

from .randomization import randomization_cmd
from .robustness import robustness_cmd

experiments_app = typer.Typer(
    name="experiments",
    help="Experiment workflow commands.",
    no_args_is_help=True,
)

experiments_app.command("robustness")(robustness_cmd)
experiments_app.command("randomization")(randomization_cmd)
