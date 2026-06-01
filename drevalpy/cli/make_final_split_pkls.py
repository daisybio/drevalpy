"""``drevalpy make-final-split-pkls`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli_model_testing import run_final_split


def register(app: typer.Typer) -> None:
    @app.command("make-final-split-pkls")
    def make_final_split_pkls(
        response: Annotated[
            str, typer.Option("--response", help="Drug response data, pickled (output of load_response).")
        ],
        model_name: Annotated[
            str,
            typer.Option("--model_name", help="Model class name, e.g., RandomForest, SingleDrugRandomForest."),
        ],
        path_data: Annotated[str, typer.Option("--path_data", help="Path to data. Default: data.")] = "data",
        test_mode: Annotated[
            str, typer.Option("--test_mode", help="Test mode (LPO, LCO, LTO, LDO). Default: LPO.")
        ] = "LPO",
        val_ratio: Annotated[float, typer.Option("--val_ratio", help="Validation ratio.")] = 0.1,
    ) -> None:
        """Create train/validation/early-stopping pickles for a final production model."""
        run_final_split(
            response=response,
            model_name=model_name,
            path_data=path_data,
            test_mode=test_mode,
            val_ratio=val_ratio,
        )
