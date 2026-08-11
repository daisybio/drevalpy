"""``drevalpy report`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath


def report_cmd(
    experiment_dir: Annotated[str, typer.Argument(help="Path to a saved ExperimentResult directory.")],
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory for the report.")] = "report",
) -> None:
    """Generate visualizations from an ExperimentResult."""
    from drevalpy.types.results import ExperimentResult
    from drevalpy.visualization.create_visualizations import create_visualizations

    exp_path = UPath(experiment_dir)
    out = UPath(output_dir)

    experiment = ExperimentResult.load(str(exp_path))
    create_visualizations(experiment, out)

    typer.echo(f"Report generated at {out}")
