"""``drevalpy consolidate-single-drug`` command."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli._helpers import as_list
from drevalpy.cli_model_testing import run_consolidate_results


def register(app: typer.Typer) -> None:
    @app.command("consolidate-single-drug")
    def consolidate_single_drug(
        run_id: Annotated[str, typer.Option("--run_id", help="Run ID")],
        model_name: Annotated[str, typer.Option("--model_name", help="All Model names")],
        outdir_path: Annotated[str, typer.Option("--outdir_path", help="Output directory path")],
        n_cv_splits: Annotated[int, typer.Option("--n_cv_splits", help="Number of CV splits")],
        test_mode: Annotated[str, typer.Option("--test_mode", help="Test mode (LPO, LCO, LTO, LDO)")] = "LPO",
        cross_study_datasets: Annotated[
            list[str] | None, typer.Option("--cross_study_datasets", help="Cross-study datasets (space-separated).")
        ] = None,
        randomization_modes: Annotated[
            str, typer.Option("--randomization_modes", help="All randomizations")
        ] = "[None]",
        n_trials_robustness: Annotated[int, typer.Option("--n_trials_robustness", help="Number of trials")] = 0,
    ) -> None:
        """Consolidate results for SingleDrugModels."""
        run_consolidate_results(
            run_id=run_id,
            test_mode=test_mode,
            model_name=model_name,
            outdir_path=outdir_path,
            n_cv_splits=n_cv_splits,
            cross_study_datasets=as_list(cross_study_datasets) if cross_study_datasets else None,
            randomization_modes=randomization_modes,
            n_trials_robustness=n_trials_robustness,
        )
