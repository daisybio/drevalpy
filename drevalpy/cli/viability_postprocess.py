"""``drevalpy viability-postprocess`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from drevalpy.cli.preprocess_custom import run_postprocess_viability

# Module-level constants so the Typer defaults are not fresh calls (flake8 B008).
_CWD = Path()


def register(app: typer.Typer) -> None:
    """Register the ``viability-postprocess`` subcommand on ``app``.

    :param app: Root Typer application to register the command on.
    """

    @app.command("viability-postprocess")
    def viability_postprocess(
        dataset_name: Annotated[
            str,
            typer.Option("--dataset_name", help="Dataset name, e.g., MyCustomDataset."),
        ],
        path_data: Annotated[
            Path,
            typer.Option(
                "--path_data",
                help="Path to output folder of CurveCurator containing the curves.txt file, default: './'.",
            ),
        ] = _CWD,
    ) -> None:
        """Postprocess CurveCurator viability data into one CSV.

        :param dataset_name: Custom dataset name.
        :param path_data: Directory containing CurveCurator output files.
        """
        run_postprocess_viability(dataset_name=dataset_name, path_data=path_data)
