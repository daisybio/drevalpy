"""``drevalpy aggregate`` command."""

from __future__ import annotations

from typing import Annotated

import typer
from upath import UPath


def aggregate_cmd(
    results: Annotated[list[str], typer.Argument(help="Paths to RunResult .npz files.")],
    output_dir: Annotated[
        str, typer.Option("--output-dir", "-o", help="Output directory for the ExperimentResult.")
    ] = "experiment_results",
) -> None:
    """Aggregate parallel RunResult files into an ExperimentResult."""
    from drevalpy.types.results import ExperimentResult, RunResult

    out = UPath(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    run_results = [RunResult.load(path) for path in results]
    experiment = ExperimentResult(run_results)
    experiment.save(str(out))

    typer.echo(f"Aggregated {len(run_results)} runs into ExperimentResult at {out}")
    typer.echo(repr(experiment))
