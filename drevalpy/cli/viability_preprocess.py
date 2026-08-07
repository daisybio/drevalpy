"""``drevalpy viability-preprocess`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from drevalpy.cli.preprocess_custom import run_preprocess_raw_viability

# Module-level constants so the Typer defaults are not fresh calls (flake8 B008).
_DEFAULT_DATA_DIR = Path("data")


def register(app: typer.Typer) -> None:
    """Register the ``viability-preprocess`` subcommand on ``app``.

    :param app: Root Typer application to register the command on.
    """

    @app.command("viability-preprocess")
    def viability_preprocess(
        dataset_name: Annotated[
            str,
            typer.Option("--dataset_name", help="Dataset name, e.g., MyCustomDataset."),
        ],
        path_data: Annotated[
            Path,
            typer.Option(
                "--path_data",
                help="Path to base folder containing datasets, in particular dataset_name/dataset_name_raw.csv, "
                "default: ./data.",
            ),
        ] = _DEFAULT_DATA_DIR,
        cores: Annotated[
            int,
            typer.Option(
                "--cores",
                help="The number of cores used for CurveCurator fitting, default: 4.",
            ),
        ] = 4,
    ) -> None:
        """Preprocess CurveCurator viability data.

        :param dataset_name: Custom dataset name.
        :param path_data: Root data directory containing raw viability CSVs.
        :param cores: Worker count for CurveCurator fitting.
        """
        run_preprocess_raw_viability(path_data=path_data, dataset_name=dataset_name, cores=cores)
