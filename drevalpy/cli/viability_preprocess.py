"""``drevalpy viability-preprocess`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli_preprocess_custom import run_preprocess_raw_viability


def register(app: typer.Typer) -> None:
    @app.command("viability-preprocess")
    def viability_preprocess(
        dataset_name: Annotated[str, typer.Option("--dataset_name", help="Dataset name, e.g., MyCustomDataset.")],
        path_data: Annotated[
            str,
            typer.Option(
                "--path_data",
                help="Path to base folder containing datasets, in particular dataset_name/dataset_name_raw.csv, "
                "default: ./data.",
            ),
        ] = "./data",
        cores: Annotated[
            int,
            typer.Option("--cores", help="The number of cores used for CurveCurator fitting, default: 4."),
        ] = 4,
    ) -> None:
        """Preprocess CurveCurator viability data."""
        run_preprocess_raw_viability(path_data=path_data, dataset_name=dataset_name, cores=cores)
