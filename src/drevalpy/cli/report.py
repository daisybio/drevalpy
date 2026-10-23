"""``drevalpy report`` command."""

from __future__ import annotations

from typing import Annotated

import typer


def report_cmd(
    experiment_dir: Annotated[str, typer.Argument(help="Path to a saved ExperimentResult directory.")],
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory for the report.")] = "report",
    title: Annotated[str, typer.Option("--title", "-t", help="Report title.")] = "Drug Response Evaluation",
    reference_model: Annotated[
        str | None, typer.Option("--reference-model", "-r", help="Normalize metrics against this model.")
    ] = None,
) -> None:
    """Generate a MultiQC report from an ExperimentResult."""
    from drevalpy.types.results import ExperimentResult
    from drevalpy.visualization.report import create_report

    # No visualization reads HPO trial predictions, which outweigh the fold predictions
    # they belong to, so the report path is the one caller that opts out of loading them.
    # The experiment is passed inline rather than through a local so ``create_report``
    # holds the only reference and can release the pre-normalization copy.
    create_report(
        ExperimentResult.load(experiment_dir, with_trials=False),
        output_dir,
        title=title,
        reference_model=reference_model,
    )
    typer.echo(f"Report generated at {output_dir}")
