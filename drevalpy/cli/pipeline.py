"""Root ``drevalpy`` command (full experiment pipeline)."""

from __future__ import annotations

from importlib.metadata import version as pkg_version
from typing import Annotated

import typer
from typer import _click

from drevalpy.cli._helpers import as_list, pipeline_namespace
from drevalpy.evaluation import AVAILABLE_METRICS
from drevalpy.utils import check_arguments, main


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"drevalpy {pkg_version('drevalpy')}")
        raise typer.Exit()


BASELINES_HELP = (
    "List of baselines to evaluate. If NaiveMeanEffectsPredictor is not part of them, we will add it."
    "The baselines are also hyperparameter-tuned and compared to the models, but no randomization or robustness tests "
    "are run. NaiveMeanEffectsPredictor is always run as it is required for evaluation."
)
TEST_MODE_HELP = (
    "Which tests to run (LPO=Leave-random-Pairs-Out, LCO=Leave-Cell-line-Out, LTO=Leave-Tissue-Out, LDO=Leave-Drug-Out)"
    ". Can be a list, e.g. 'LPO LCO LTO LDO' to run all tests. Default is LPO."
)
RANDOMIZATION_MODE_HELP = (
    "Which randomization tests to run, additionally to the normal run. None disables randomization tests. "
    "Available modes: SVCC, SVRC, SVCD, SVRD. Can be a list of randomization tests, "
    "e.g. 'SVCC SVCD'. SVCC - Single View Constant (while others are perturbed) for Cell Lines, "
    "SVRC - Single View Random (while others are held constant) for Cell Lines, "
    "SVCD - Single View Constant for Drugs, SVRD - Single View Random for Drugs."
)
RANDOMIZATION_TYPE_HELP = (
    'Type of randomization to use. Choose from "permutation" or "invariant". Default is "permutation". '
    "Permutation shuffles features over instances while preserving feature distributions. Invariant "
    "randomization preserves a key characteristic such as matrix mean and standard deviation or network degree."
)
ROBUSTNESS_HELP = (
    "Number of trials to run for the robustness test. Default is 0, which means no robustness test is run. "
    "The robustness test trains the model with varying seeds multiple times to check stability."
)
NO_REFITTING_HELP = (
    "If not set, the measure is appended with '_curvecurator'. If a custom dataset_name was provided, this will invoke "
    "the fitting procedure of raw viability data, which is expected to exist at "
    "``<path_data>/<dataset_name>/<dataset_name>_raw.csv``. The fitted dataset will be stored in the same folder, "
    "in a file called ``<dataset_name>.csv``. Default is False, i.e., curvecurated drug response measures are utilized."
)

MEASURE_HELP = (
    "Drug response measure used as prediction target. If using one of the available datasets, this is restricted to "
    "one of ['LN_IC50', 'EC50', 'IC50', 'pEC50', 'AUC', 'response']. This corresponds to the names of the columns that "
    "contain theses measures in the provided input dataset. If providing a custom dataset, this may differ. "
    "If the option ``--no_refitting`` is not set, the prefix '_curvecurator' is automatically appended, "
    "e.g., 'LN_IC50_curvecurator', to allow using the refit measures instead of the ones originally published for the "
    "available datasets, allowing for better dataset comparability (refit measures are already provided in the "
    "available datasets or computed as part of the fitting procedure when providing custom raw viability datasets, "
    "see ``--no_refitting`` for details). Default: ``LN_IC50``"
)

RESPONSE_TRANSFORMATION_HELP = (
    "Transformation to apply to the response variable during training and prediction. Will be retransformed "
    "after the final predictions. Possible values: standard, minmax, robust."
)


def register_pipeline_callback(app: typer.Typer) -> None:
    """Register the default callback that runs the full pipeline when no subcommand is given."""

    @app.callback(invoke_without_command=True)
    def pipeline_root(
        ctx: _click.Context,
        show_version: Annotated[
            bool,
            typer.Option(
                "--version",
                "-v",
                callback=_version_callback,
                is_eager=True,
                help="Show version and exit.",
                expose_value=False,
            ),
        ] = False,
        run_id: Annotated[str, typer.Option("--run_id", help="Identifier to save the results.")] = "my_run",
        path_data: Annotated[str, typer.Option("--path_data", help="Path to the data directory.")] = "data",
        models: Annotated[
            list[str] | None, typer.Option("--models", help="Model to evaluate or list of models to compare.")
        ] = None,
        baselines: Annotated[list[str] | None, typer.Option("--baselines", help=BASELINES_HELP)] = None,
        test_mode: Annotated[list[str] | None, typer.Option("--test_mode", help=TEST_MODE_HELP)] = None,
        randomization_mode: Annotated[
            list[str] | None, typer.Option("--randomization_mode", help=RANDOMIZATION_MODE_HELP)
        ] = None,
        randomization_type: Annotated[str, typer.Option("--randomization_type", help=RANDOMIZATION_TYPE_HELP)] = (
            "permutation"
        ),
        n_trials_robustness: Annotated[int, typer.Option("--n_trials_robustness", help=ROBUSTNESS_HELP)] = 0,
        dataset_name: Annotated[str, typer.Option("--dataset_name", help="Name of the dataset to use.")] = ("GDSC1"),
        cross_study_datasets: Annotated[
            list[str] | None,
            typer.Option(
                "--cross_study_datasets",
                help="List of datasets to use for cross-study prediction evaluation. Default is empty list.",
            ),
        ] = None,
        path_out: Annotated[str, typer.Option("--path_out", help="Path to the output directory.")] = "results/",
        no_refitting: Annotated[bool, typer.Option("--no_refitting", help=NO_REFITTING_HELP)] = False,
        curve_curator_cores: Annotated[
            int,
            typer.Option(
                "--curve_curator_cores",
                help="Max. number of cores used to fit curves with CurveCurator following min(cores, #curves to fit).",
            ),
        ] = 1,
        curve_curator_normalize: Annotated[
            bool,
            typer.Option(
                "--curve_curator_normalize",
                help="Whether to normalize the response values to [0, 1] for CurveCurator. Default is False.",
            ),
        ] = False,
        measure: Annotated[
            str,
            typer.Option(
                "--measure",
                help=MEASURE_HELP,
            ),
        ] = "LN_IC50",
        overwrite: Annotated[
            bool, typer.Option("--overwrite", help="Overwrite existing results with the same path out and run_id?")
        ] = False,
        optim_metric: Annotated[
            str,
            typer.Option(
                "--optim_metric",
                help=f"Metric for hyperparameter tuning choose from {list(AVAILABLE_METRICS.keys())}. Default is RMSE.",
            ),
        ] = "RMSE",
        wandb_project: Annotated[
            str | None,
            typer.Option(
                "--wandb_project",
                help=(
                    "Optional Weights & Biases project name. If provided, enables wandb logging for all DRPModel "
                    "instances."
                ),
            ),
        ] = None,
        n_cv_splits: Annotated[
            int, typer.Option("--n_cv_splits", help="Number of cross-validation splits to use for the evaluation.")
        ] = 7,
        response_transformation: Annotated[
            str, typer.Option("--response_transformation", help=RESPONSE_TRANSFORMATION_HELP)
        ] = "None",
        multiprocessing: Annotated[
            bool,
            typer.Option("--multiprocessing", help="If set, we will use raytune for fitting. Default is False."),
        ] = False,
        model_checkpoint_dir: Annotated[
            str, typer.Option("--model_checkpoint_dir", help="Directory to save model checkpoints.")
        ] = "TEMPORARY",
        final_model_on_full_data: Annotated[
            bool,
            typer.Option(
                "--final_model_on_full_data",
                help="Save a final model trained and tuned on the union of all folds after cross-validation.",
            ),
        ] = False,
        no_hyperparameter_tuning: Annotated[
            bool,
            typer.Option(
                "--no_hyperparameter_tuning", help="Disable hyperparameter tuning and use first hyperparameter set."
            ),
        ] = False,
    ) -> None:
        """Run the drug response prediction model test suite."""
        if ctx.invoked_subcommand is not None:
            return
        args = pipeline_namespace(
            run_id=run_id,
            path_data=path_data,
            models=as_list(models),
            baselines=as_list(baselines) if baselines is not None else None,
            test_mode=as_list(test_mode) if test_mode is not None else ["LPO"],
            randomization_mode=as_list(randomization_mode) if randomization_mode is not None else ["None"],
            randomization_type=randomization_type,
            n_trials_robustness=n_trials_robustness,
            dataset_name=dataset_name,
            cross_study_datasets=as_list(cross_study_datasets),
            path_out=path_out,
            no_refitting=no_refitting,
            curve_curator_cores=curve_curator_cores,
            curve_curator_normalize=curve_curator_normalize,
            measure=measure,
            overwrite=overwrite,
            optim_metric=optim_metric,
            wandb_project=wandb_project,
            n_cv_splits=n_cv_splits,
            response_transformation=response_transformation,
            multiprocessing=multiprocessing,
            model_checkpoint_dir=model_checkpoint_dir,
            final_model_on_full_data=final_model_on_full_data,
            no_hyperparameter_tuning=no_hyperparameter_tuning,
        )
        check_arguments(args)
        main(args)
