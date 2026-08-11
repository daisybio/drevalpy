"""``drevalpy report`` command."""

from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from drevalpy.data._paths import get_default_data_dir
from drevalpy.visualization._legacy.create_report import run_report

# Module-level constants so the Typer defaults are not fresh calls (flake8 B008).
_DEFAULT_RESULTS_DIR = pathlib.Path("results")


def register(app: typer.Typer) -> None:
    """Register the ``report`` subcommand on ``app``.

    :param app: Root Typer application to register the command on.
    """

    @app.command("report")
    def report(
        run_id: Annotated[str, typer.Option("--run_id", help="Run ID for the current execution")],
        dataset_name: Annotated[
            str, typer.Option("--dataset_name", help="Name of the dataset for which to render the result file")
        ],
        result_path: Annotated[
            pathlib.Path, typer.Option("--result_path", help="Path to the results")
        ] = _DEFAULT_RESULTS_DIR,
    ) -> None:
        """Generate reports from evaluation results.

        :param run_id: Experiment run identifier.
        :param dataset_name: Dataset name used to locate result files.
        :param result_path: Directory containing experiment outputs.
        """
        run_report(run_id=run_id, dataset=dataset_name, path_data=get_default_data_dir(), result_path=result_path)
