"""Typer entry point for the ``drevalpy`` console script."""

from __future__ import annotations

import typer

from drevalpy.cli.data import data_app
from drevalpy.cli.experiments import experiments_app

app = typer.Typer(
    name="drevalpy",
    help="Drug response evaluation framework.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(data_app, name="data")
app.add_typer(experiments_app, name="experiments")


def cli_main() -> None:
    """Console script entry point."""
    app()


if __name__ == "__main__":
    cli_main()
