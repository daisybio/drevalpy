"""``drevalpy train-final-model`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli_model_testing import run_train_final_model


def register(app: typer.Typer) -> None:
    @app.command("train-final-model")
    def train_final_model_cmd(
        train_data: Annotated[str, typer.Option("--train_data", help="Train data, pickled.")],
        val_data: Annotated[str, typer.Option("--val_data", help="Validation data, pickled.")],
        early_stopping_data: Annotated[
            str, typer.Option("--early_stopping_data", help="Early stopping data, pickled.")
        ],
        model_name: Annotated[
            str,
            typer.Option(
                "--model_name",
                help="Model name (model_name for global models, model_name.drug_name for single-drug models).",
            ),
        ],
        best_hpam_combi: Annotated[
            str, typer.Option("--best_hpam_combi", help="Best hyperparameter combination file, yaml format.")
        ],
        path_data: Annotated[str, typer.Option("--path_data", help="Path to data. Default: data.")] = "data",
        response_transformation: Annotated[
            str, typer.Option("--response_transformation", help="Response transformation.")
        ] = "None",
        model_checkpoint_dir: Annotated[
            str,
            typer.Option(
                "--model_checkpoint_dir",
                help="model checkpoint directory, if not provided: temporary directory is used",
            ),
        ] = "TEMPORARY",
    ) -> None:
        """Train a final model on the full dataset using the best hyperparameters."""
        run_train_final_model(
            train_data=train_data,
            val_data=val_data,
            early_stopping_data=early_stopping_data,
            response_transformation=response_transformation,
            model_name=model_name,
            path_data=path_data,
            model_checkpoint_dir=model_checkpoint_dir,
            best_hpam_combi=best_hpam_combi,
        )
