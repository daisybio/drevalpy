"""``drevalpy viability-preprocess`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli.preprocess_custom import run_preprocess_raw_viability
from drevalpy.data._paths import get_default_data_dir


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
        :param cores: Worker count for CurveCurator fitting.
        """
        run_preprocess_raw_viability(path_data=get_default_data_dir(), dataset_name=dataset_name, cores=cores)
