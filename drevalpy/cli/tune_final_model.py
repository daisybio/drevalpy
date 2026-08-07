"""``drevalpy tune-final-model`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from drevalpy.cli.model_testing import run_tune_final_model

# Module-level constants so the Typer defaults are not fresh calls (flake8 B008).
_DEFAULT_DATA_DIR = Path("data")


def register(app: typer.Typer) -> None:
    """Register the ``tune-final-model`` subcommand on ``app``.

    :param app: Root Typer application to register the command on.
    """

    @app.command("tune-final-model")
    def tune_final_model(
        train_data: Annotated[Path, typer.Option("--train_data", help="Train dataset, pickled.")],
        val_data: Annotated[Path, typer.Option("--val_data", help="Validation dataset, pickled.")],
        early_stopping_data: Annotated[
            str,
            typer.Option("--early_stopping_data", help="Early stopping dataset, pickled."),
        ],
        model_name: Annotated[
            str,
            typer.Option(
                "--model_name",
                help="Model name (model_name for global models, model_name.drug_name for single-drug models).",
            ),
        ],
        hpam_combi: Annotated[
            str,
            typer.Option(
                "--hpam_combi",
                help="Path to hyperparameter combination file, yaml format.",
            ),
        ],
        path_data: Annotated[
            Path, typer.Option("--path_data", help="Path to data. Default: data.")
        ] = _DEFAULT_DATA_DIR,
        response_transformation: Annotated[
            str,
            typer.Option(
                "--response_transformation",
                help="Response transformation. Default: None.",
            ),
        ] = "None",
        model_checkpoint_dir: Annotated[
            Path | None,
            typer.Option(
                "--model_checkpoint_dir",
                help="model checkpoint directory, if not provided: temporary directory is used",
            ),
        ] = None,
    ) -> None:
        """Score one hyperparameter YAML on the final validation split.

        This does not run Ray/Optuna search. Prefer the root experiment or
        ``drevalpy.experiment.train_final_model`` for real tuning.

        :param train_data: Path to pickled training dataset.
        :param val_data: Path to pickled validation dataset.
        :param early_stopping_data: Path to pickled early-stopping dataset.
        :param model_name: Registered model name.
        :param hpam_combi: Path to a YAML hyperparameter file.
        :param path_data: Root data directory passed to feature loaders.
        :param response_transformation: Sklearn response transform name.
        :param model_checkpoint_dir: Directory for model checkpoints, or ``None`` for a temporary one.
        """
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
