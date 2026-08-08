"""``drevalpy test-cv`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from drevalpy.cli._helpers import as_list
from drevalpy.cli.model_testing import run_train_and_predict_final


def register(app: typer.Typer) -> None:
    """Register the ``test-cv`` subcommand on ``app``.

    :param app: Root Typer application to register the command on.
    """

    @app.command("test-cv")
    def test_cv(
        model_name: Annotated[
            str,
            typer.Option(
                "--model_name",
                help="Model name for global models, <Model name>.<Drug name> for single-drug models.",
            ),
        ],
        split_id: Annotated[str, typer.Option("--split_id", help="Split id.")],
        split_dataset_path: Annotated[
            str,
            typer.Option("--split_dataset_path", help="Path to the pickled CV split dataset."),
        ],
        hyperparameters_path: Annotated[
            str,
            typer.Option(
                "--hyperparameters_path",
                help="Path to yaml file containing the optimal hyperparameters.",
            ),
        ],
        mode: Annotated[
            str,
            typer.Option(
                "--mode",
                help="Mode: full, randomization, or robustness. Default: full.",
            ),
        ] = "full",
        response_transformation: Annotated[
            str,
            typer.Option(
                "--response_transformation",
                help="Response transformation. Default: None.",
            ),
        ] = "None",
        test_mode: Annotated[
            str,
            typer.Option("--test_mode", help="Test mode (LPO, LCO, LTO, LDO). Default: LPO."),
        ] = "LPO",
        randomization_views_path: Annotated[
            str | None,
            typer.Option(
                "--randomization_views_path",
                help="Path to the yaml file containing the randomization configuration "
                "(only relevant if mode=randomization).",
            ),
        ] = None,
        randomization_type: Annotated[
            str,
            typer.Option(
                "--randomization_type",
                help="Randomization type (permutation, invariant). Default: permutation. "
                "Only relevant if mode=randomization.",
            ),
        ] = "permutation",
        robustness_trial: Annotated[
            int | None,
            typer.Option(
                "--robustness_trial",
                help="Robustness trial index. Only relevant if mode=robustness.",
            ),
        ] = None,
        cross_study_datasets: Annotated[
            list[str] | None,
            typer.Option(
                "--cross_study_datasets",
                help="Paths to pickled cross study datasets (space-separated).",
            ),
        ] = None,
        model_checkpoint_dir: Annotated[
            Path | None,
            typer.Option(
                "--model_checkpoint_dir",
                help="model checkpoint directory, if not provided: temporary directory is used",
            ),
        ] = None,
    ) -> None:
        """Train and predict on the CV test set (full, randomization, or robustness mode).

        :param model_name: Registered model name (optionally ``Model.drug`` for single-drug).
        :param split_id: CV split identifier used in output paths.
        :param split_dataset_path: Path to a pickled CV split artifact.
        :param hyperparameters_path: Path to the best-hyperparameter YAML for this split.
        :param mode: One of ``full``, ``randomization``, or ``robustness``.
        :param response_transformation: Sklearn response transform name.
        :param test_mode: Split label passed to cross-study prediction.
        :param randomization_views_path: YAML path when ``mode`` is ``randomization``.
        :param randomization_type: ``permutation`` or ``invariant`` for randomization mode.
        :param robustness_trial: Trial index when ``mode`` is ``robustness``.
        :param cross_study_datasets: Optional pickled cross-study datasets for full mode.
        :param model_checkpoint_dir: Directory for model checkpoints, or ``None`` for a temporary one.
        """
        run_train_and_predict_final(
            mode=mode,
            model_name=model_name,
            split_id=split_id,
            split_dataset_path=split_dataset_path,
            hyperparameters_path=hyperparameters_path,
            response_transformation=response_transformation,
            test_mode=test_mode,
            randomization_views_path=randomization_views_path,
            randomization_type=randomization_type,
            robustness_trial=robustness_trial,
            cross_study_datasets=as_list(cross_study_datasets) if cross_study_datasets else None,
            model_checkpoint_dir=model_checkpoint_dir,
        )
