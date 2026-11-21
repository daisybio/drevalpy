"""``drevalpy aggregate`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath

from ._options import GroupByOption
from ._summary import format_grouped_summary, resolve_group_by_dataset


def aggregate_cmd(
    results: Annotated[list[str], typer.Argument(help="Paths to RunResult .npz files.")],
    output_dir: Annotated[
        str, typer.Option("--output-dir", "-o", help="Output directory for the ExperimentResult.")
    ] = "experiment_results",
    group_by: GroupByOption = None,
    dataset: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Dataset name or .h5mu path the runs were made on; repeat for several studies. "
            "Read only for a --group-by that is a dataset annotation.",
        ),
    ] = None,
) -> None:
    """Aggregate parallel RunResult files into an ExperimentResult."""
    from drevalpy.types.results import ExperimentResult, RunResult

    group_dataset = resolve_group_by_dataset(group_by, dataset)
    run_results = [RunResult.load(path) for path in results]
    experiment = ExperimentResult(run_results)
    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    experiment.save(str(out))

    typer.echo(f"Aggregated {len(run_results)} runs into ExperimentResult at {out}")
    typer.echo(repr(experiment))
    if group_by is not None:
        typer.echo(format_grouped_summary(experiment, group_by, group_dataset))
