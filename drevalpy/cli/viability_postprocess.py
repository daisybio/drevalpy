"""``drevalpy viability-postprocess`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli_preprocess_custom import run_postprocess_viability


def register(app: typer.Typer) -> None:
    @app.command("viability-postprocess")
    def viability_postprocess(
        dataset_name: Annotated[str, typer.Option("--dataset_name", help="Dataset name, e.g., MyCustomDataset.")],
        path_data: Annotated[
            str,
            typer.Option(
                "--path_data",
                help="Path to output folder of CurveCurator containing the curves.txt file, default: './'.",
            ),
        ] = "./",
    ) -> None:
        """Postprocess CurveCurator viability data into one CSV."""
        run_postprocess_viability(dataset_name=dataset_name, path_data=path_data)
