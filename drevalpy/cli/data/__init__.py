"""``drevalpy data`` command group."""

from __future__ import annotations

import typer

from drevalpy.cli.data.load import load_dataset
from drevalpy.cli.data.split import split_dataset

data_app = typer.Typer(
    name="data",
    help="Data management commands.",
    no_args_is_help=True,
)

data_app.command("load")(load_dataset)
data_app.command("split")(split_dataset)
