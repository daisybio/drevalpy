"""``drevalpy viability-postprocess`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli.preprocess_custom import run_postprocess_viability
from drevalpy.data._paths import get_default_data_dir


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
    ) -> None:
        """Postprocess CurveCurator viability data into one CSV.

        :param dataset_name: Custom dataset name.
        """
        run_postprocess_viability(dataset_name=dataset_name, path_data=get_default_data_dir())
