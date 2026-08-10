"""``drevalpy train-final-model`` command."""

from __future__ import annotations

import pathlib
from typing import Annotated

import typer

from drevalpy.cli.model_testing import run_train_final_model


def register(app: typer.Typer) -> None:
    """Register the ``train-final-model`` subcommand on ``app``.

    :param app: Root Typer application to register the command on.
    """

    @app.command("train-final-model")
    def train_final_model_cmd(
        train_data: Annotated[pathlib.Path, typer.Option("--train_data", help="Train data, pickled.")],
        val_data: Annotated[pathlib.Path, typer.Option("--val_data", help="Validation data, pickled.")],
        early_stopping_data: Annotated[
            str,
            typer.Option("--early_stopping_data", help="Early stopping data, pickled."),
        ],
        model_name: Annotated[
            str,
            typer.Option(
                "--model_name",
                help="Model name (model_name for global models, model_name.drug_name for single-drug models).",
            ),
        ],
        best_hpam_combi: Annotated[
            pathlib.Path,
            typer.Option(
                "--best_hpam_combi",
                help="Best hyperparameter combination file, yaml format.",
            ),
        ],
        response_transformation: Annotated[
            str,
            typer.Option("--response_transformation", help="Response transformation."),
        ] = "None",
        model_checkpoint_dir: Annotated[
            pathlib.Path | None,
            typer.Option(
                "--model_checkpoint_dir",
                help="model checkpoint directory, if not provided: temporary directory is used",
            ),
        ] = None,
    ) -> None:
        """Train a final model on the full dataset using the best hyperparameters.

        :param train_data: Path to pickled training dataset.
        :param val_data: Path to pickled validation dataset.
        :param early_stopping_data: Path to pickled early-stopping dataset.
        :param model_name: Registered model name.
        :param best_hpam_combi: Path to YAML with the selected hyperparameters.
        :param response_transformation: Sklearn response transform name.
        :param model_checkpoint_dir: Directory for model checkpoints, or ``None`` for a temporary one.
        """
        run_train_final_model(
            train_data=train_data,
            val_data=val_data,
            early_stopping_data=early_stopping_data,
            response_transformation=response_transformation,
            model_name=model_name,
            model_checkpoint_dir=model_checkpoint_dir,
            best_hpam_combi=best_hpam_combi,
        )
