"""``drevalpy tune-final-model`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli_model_testing import run_tune_final_model


def register(app: typer.Typer) -> None:
    @app.command("tune-final-model")
    def tune_final_model(
        train_data: Annotated[str, typer.Option("--train_data", help="Train dataset, pickled.")],
        val_data: Annotated[str, typer.Option("--val_data", help="Validation dataset, pickled.")],
        early_stopping_data: Annotated[
            str, typer.Option("--early_stopping_data", help="Early stopping dataset, pickled.")
        ],
        model_name: Annotated[
            str,
            typer.Option(
                "--model_name",
                help="Model name (model_name for global models, model_name.drug_name for single-drug models).",
            ),
        ],
        hpam_combi: Annotated[
            str, typer.Option("--hpam_combi", help="Path to hyperparameter combination file, yaml format.")
        ],
        path_data: Annotated[str, typer.Option("--path_data", help="Path to data. Default: data.")] = "data",
        response_transformation: Annotated[
            str, typer.Option("--response_transformation", help="Response transformation. Default: None.")
        ] = "None",
        model_checkpoint_dir: Annotated[
            str,
            typer.Option(
                "--model_checkpoint_dir",
                help="model checkpoint directory, if not provided: temporary directory is used",
            ),
        ] = "TEMPORARY",
    ) -> None:
        """Find optimal hyperparameters for the final model on full data."""
        run_tune_final_model(
            train_data=train_data,
            val_data=val_data,
            early_stopping_data=early_stopping_data,
            model_name=model_name,
            hpam_combi=hpam_combi,
            response_transformation=response_transformation,
            path_data=path_data,
            model_checkpoint_dir=model_checkpoint_dir,
        )
