"""Typer entry point for the ``drevalpy`` console script."""

from __future__ import annotations

import logging
from typing import Annotated, Any

import typer
from typer.core import TyperGroup

from drevalpy.cli.aggregate import aggregate_cmd
from drevalpy.cli.catalog import list_app
from drevalpy.cli.curate import curate_cmd
from drevalpy.cli.data import data_app
from drevalpy.cli.experiments import experiments_app
from drevalpy.cli.report import report_cmd
from drevalpy.cli.run import run_cmd
from drevalpy.cli.single import single_cmd
from drevalpy.errors import DrEvalError


class _DrEvalGroup(TyperGroup):
    """Root command group that reports user mistakes as one-line errors instead of tracebacks."""

    def invoke(self, ctx: typer.Context) -> Any:
        """Run the callback and subcommand, printing drevalpy input errors as one line.

        :param ctx: Context of the root group.
        :returns: Whatever the invoked command returns.
        :raises typer.Exit: With code 1 after ``Error: <message>`` went to stderr, for a
            :class:`~drevalpy.errors.DrEvalError` or a ``FileNotFoundError``. Other exceptions propagate.
        """
        try:
            return super().invoke(ctx)
        except (DrEvalError, FileNotFoundError) as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(1) from error


app = typer.Typer(
    cls=_DrEvalGroup,
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
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Log debug detail, such as per-model timings."),
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Log warnings and errors only.")] = False,
) -> None:
    """Global options applied before any subcommand.

    :param extensions_dir: Directories with extension files to load.
    :param verbose: Log at DEBUG instead of INFO.
    :param quiet: Log at WARNING instead of INFO.
    :raises typer.BadParameter: If both *verbose* and *quiet* are given.
    """
    from drevalpy.log import setup_logging
    from drevalpy.registry import load_extensions

    if verbose and quiet:
        msg = "--verbose and --quiet are mutually exclusive."
        raise typer.BadParameter(msg)
    setup_logging(logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO)
    load_extensions()
    if extensions_dir:
        load_extensions(directories=extensions_dir)


app.add_typer(data_app, name="data")
app.add_typer(experiments_app, name="experiments")
app.add_typer(list_app, name="list")
app.command("run")(run_cmd)
app.command("single")(single_cmd)
app.command("aggregate")(aggregate_cmd)
app.command("curate")(curate_cmd)
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
