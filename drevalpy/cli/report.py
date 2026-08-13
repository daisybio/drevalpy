"""``drevalpy report`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath

from drevalpy.log import get_logger

logger = get_logger(__name__)


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

    if dataset_path:
        # Accepted for pipeline compatibility but deliberately not read: every
        # visualization takes `dataset` and ignores it, and the .h5mu is large enough that
        # loading it eats a meaningful fraction of the report container's memory.
        logger.info("Ignoring --dataset %s: no visualization consumes dataset metadata", dataset_path)

    # No visualization reads HPO trial predictions, which outweigh the fold predictions
    # they belong to, so the report path is the one caller that opts out of loading them.
    # The experiment is passed inline rather than through a local so ``create_report``
    # holds the only reference and can release the pre-normalization copy.
    create_report(
        ExperimentResult.load(str(exp_path), with_trials=False),
        output_dir,
        title=title,
        reference_model=reference_model,
        dataset=None,
    )
    typer.echo(f"Report generated at {output_dir}")
