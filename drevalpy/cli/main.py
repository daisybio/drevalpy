"""Typer entry point for the ``drevalpy`` console script."""

from __future__ import annotations

import typer

from drevalpy.cli.data import data_app
from drevalpy.cli.experiments import experiments_app
from drevalpy.cli.run import run_cmd
from drevalpy.cli.single import single_cmd

app = typer.Typer(
    name="drevalpy",
    help="Drug response evaluation framework.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(data_app, name="data")
app.add_typer(experiments_app, name="experiments")
app.command("run")(run_cmd)
app.command("single")(single_cmd)


def cli_main() -> None:
    """Console script entry point."""
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("\nInterrupted.", err=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    cli_main()
