"""``drevalpy consolidate-single-drug`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli._helpers import as_list
from drevalpy.cli.model_testing import run_consolidate_results


def register(app: typer.Typer) -> None:
    """Register the ``consolidate-single-drug`` subcommand on ``app``.

    :param app: Root Typer application to register the command on.
    """

    @app.command("consolidate-single-drug")
    def consolidate_single_drug(
        run_id: Annotated[str, typer.Option("--run_id", help="Run ID")],
        model_name: Annotated[str, typer.Option("--model_name", help="All Model names")],
        outdir_path: Annotated[str, typer.Option("--outdir_path", help="Output directory path")],
        n_cv_splits: Annotated[int, typer.Option("--n_cv_splits", help="Number of CV splits")],
        dataset_name: Annotated[str, typer.Option("--dataset_name", help="Response dataset name")],
        test_mode: Annotated[str, typer.Option("--test_mode", help="Test mode (LPO, LCO, LTO, LDO)")] = "LPO",
        cross_study_datasets: Annotated[
            list[str] | None,
            typer.Option("--cross_study_datasets", help="Cross-study datasets (space-separated)."),
        ] = None,
        randomization_modes: Annotated[
            str, typer.Option("--randomization_modes", help="All randomizations")
        ] = "[None]",
        n_trials_robustness: Annotated[int, typer.Option("--n_trials_robustness", help="Number of trials")] = 0,
    ) -> None:
        """Consolidate results for SingleDrugModels.

        :param run_id: Experiment run identifier.
        :param model_name: Registered model name.
        :param outdir_path: Base directory containing experiment outputs.
        :param n_cv_splits: Number of CV folds to consolidate.
        :param dataset_name: Dataset name used to locate result files.
        :param test_mode: Split label used in result paths.
        :param cross_study_datasets: Optional cross-study dataset names.
        :param randomization_modes: Serialized list of randomization modes or ``[None]``.
        :param n_trials_robustness: Number of robustness trials to include.
        """
        run_consolidate_results(
            run_id=run_id,
            test_mode=test_mode,
            model_name=model_name,
            outdir_path=outdir_path,
            n_cv_splits=n_cv_splits,
            dataset_name=dataset_name,
            cross_study_datasets=as_list(cross_study_datasets) if cross_study_datasets else None,
            randomization_modes=randomization_modes,
            n_trials_robustness=n_trials_robustness,
        )
