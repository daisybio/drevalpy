"""Typer entry point for the ``drevalpy`` console script."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli.aggregate import aggregate_cmd
from drevalpy.cli.data import data_app
from drevalpy.cli.experiments import experiments_app
from drevalpy.cli.report import report_cmd
from drevalpy.cli.run import run_cmd
from drevalpy.cli.single import single_cmd

app = typer.Typer(
    name="drevalpy",
    help="Drug response evaluation framework.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def main_callback(
    extensions_dir: Annotated[
        list[str] | None,
        typer.Option("--extensions-dir", "-e", help="Directory with .py/.yaml extension files."),
    ] = None,
) -> None:
    """Global options applied before any subcommand."""
    import os

    from drevalpy.registry import load_extension_dir

    env_dir = os.environ.get("DREVALPY_EXTENSIONS_DIR")
    if env_dir:
        load_extension_dir(env_dir)

    for d in extensions_dir or []:
        load_extension_dir(d)


app.add_typer(data_app, name="data")
app.add_typer(experiments_app, name="experiments")
app.command("run")(run_cmd)
app.command("single")(single_cmd)
app.command("aggregate")(aggregate_cmd)
app.command("report")(report_cmd)


def cli_main() -> None:
    """Console script entry point."""
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("\nInterrupted.", err=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    cli_main()
