"""``drevalpy report`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath


def report_cmd(
    experiment_dir: Annotated[str, typer.Argument(help="Path to a saved ExperimentResult directory.")],
    output_dir: Annotated[str, typer.Option("--output-dir", "-o", help="Output directory for the report.")] = "report",
    title: Annotated[str, typer.Option("--title", "-t", help="Report title.")] = "Drug Response Evaluation",
    reference_model: Annotated[
        str | None, typer.Option("--reference-model", "-r", help="Normalize metrics against this model.")
    ] = None,
    dataset_path: Annotated[
        str | None, typer.Option("--dataset", "-d", help="Path to dataset .h5mu for metadata enrichment.")
    ] = None,
) -> None:
    """Generate a MultiQC report from an ExperimentResult."""
    from drevalpy.types.results import ExperimentResult
    from drevalpy.visualization.report import create_report

    exp_path = UPath(experiment_dir)
    experiment = ExperimentResult.load(str(exp_path))

    ds = None
    if dataset_path:
        from drevalpy.types.data.dataset import Dataset

        ds = Dataset.load(dataset_path)

    create_report(experiment, output_dir, title=title, reference_model=reference_model, dataset=ds)
    typer.echo(f"Report generated at {output_dir}")
