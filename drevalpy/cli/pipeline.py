"""Root ``drevalpy`` command (full experiment pipeline)."""

from __future__ import annotations

from typing import Annotated

import typer

from drevalpy.cli._helpers import as_list, pipeline_namespace
from drevalpy.evaluation import AVAILABLE_METRICS
from drevalpy.utils import check_arguments, main

BASELINES_HELP = (
    "Baseline or list of baselines. The baselines are also hpam-tuned and compared to the models, "
    "but no randomization or robustness tests are run. NaiveMeanEffectsPredictor is always run as it is "
    "required for evaluation."
)
TEST_MODE_HELP = (
    "Which tests to run (LPO=Leave-random-Pairs-Out, LCO=Leave-Cell-line-Out, LDO=Leave-Drug-Out). "
    "Can be a list of test runs, e.g. 'LPO LCO LDO' to run all tests. Default is LPO."
)
RANDOMIZATION_MODE_HELP = (
    "Which randomization tests to run, additionally to the normal run. Default is None, which means no "
    "randomization tests are run. Modes: SVCC, SVRC, SVCD, SVRD. Can be a list of randomization tests, "
    "e.g. 'SVCC SVCD' to run two tests. SVCC/SVRC randomize or hold constant cell line views; SVCD/SVRD "
    "randomize or hold constant drug views."
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
    "Whether to run CurveCurator to sort out non-reactive curves. By default, CurveCurator is applied and "
    "curve-curated metrics are used."
)
RESPONSE_TRANSFORMATION_HELP = (
    "Transformation to apply to the response variable during training and prediction. Will be retransformed "
    "after the final predictions. Possible values: standard, minmax, robust."
)
CURVE_CURATOR_DEVICE_HELP = (
    'PyTorch device for CurveCurator fitting: "auto" (CUDA, then MPS, then CPU when enough curves), '
    '"cpu", "cuda", "cuda:0", or "mps".'
)


def register_pipeline_callback(app: typer.Typer) -> None:
    """Register the default callback that runs the full pipeline when no subcommand is given."""

    @app.callback(invoke_without_command=True)
    def pipeline_root(
        ctx: typer.Context,
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
        dataset_name: Annotated[str, typer.Option("--dataset_name", help="Name of the drug response dataset.")] = (
            "GDSC1"
        ),
        cross_study_datasets: Annotated[
            list[str] | None,
            typer.Option(
                "--cross_study_datasets",
                help="List of datasets to use to evaluate predictions across studies. Default is empty list.",
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
        curve_curator_device: Annotated[
            str, typer.Option("--curve_curator_device", help=CURVE_CURATOR_DEVICE_HELP)
        ] = "auto",
        curve_curator_chunk_size: Annotated[
            int, typer.Option("--curve_curator_chunk_size", help="Maximum number of curves per CPU chunk.")
        ] = 1_000,
        curve_curator_gpu_min_curves: Annotated[
            int,
            typer.Option(
                "--curve_curator_gpu_min_curves",
                help="Minimum curves in a chunk before auto device selection may use an accelerator.",
            ),
        ] = 1_000,
        curve_curator_gpu_chunk_size: Annotated[
            int,
            typer.Option(
                "--curve_curator_gpu_chunk_size",
                help="Maximum number of curves per accelerator chunk.",
            ),
        ] = 50_000,
        measure: Annotated[
            str,
            typer.Option(
                "--measure",
                help="The drug response measure used as prediction target. Can be one of ['LN_IC50', 'response'].",
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
            typer.Option(
                "--multiprocessing", help="Whether to use multiprocessing for the evaluation. Default is False."
            ),
        ] = False,
        model_checkpoint_dir: Annotated[
            str, typer.Option("--model_checkpoint_dir", help="Directory to save model checkpoints.")
        ] = "TEMPORARY",
        final_model_on_full_data: Annotated[
            bool,
            typer.Option(
                "--final_model_on_full_data",
                help="If True, saves a final model, trained/tuned on the union of all folds after CV.",
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
            curve_curator_device=curve_curator_device,
            curve_curator_chunk_size=curve_curator_chunk_size,
            curve_curator_gpu_min_curves=curve_curator_gpu_min_curves,
            curve_curator_gpu_chunk_size=curve_curator_gpu_chunk_size,
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
