"""Main evaluation pipeline entry and dataset loading helpers."""

from drevalpy.datasets.dataset import DrugResponseDataset
from drevalpy.datasets.loader import load_dataset
from drevalpy.experiment import drug_response_experiment
from drevalpy.models._model_lookup import get_model_class

from .response_transform import get_response_transformation
from .validation import check_arguments


def main(args) -> None:
    """Run the drug response evaluation pipeline.

    Args:
        args: Parsed command-line arguments for the evaluation pipeline.
    """
    check_arguments(args)
    response_data, cross_study_datasets = get_datasets(
        dataset_name=args.dataset_name,
        cross_study_datasets=args.cross_study_datasets,
        path_data=args.path_data,
        measure=args.measure,
        curve_curator=(not args.no_refitting),
        cores=args.curve_curator_cores,
        normalize=getattr(args, "curve_curator_normalize", False),
    )

    models = [get_model_class(model) for model in args.models]

    if args.baselines is not None:
        baselines = [get_model_class(baseline) for baseline in args.baselines]
    else:
        baselines = []

    if args.randomization_mode[0] == "None":
        args.randomization_mode = None
    response_transformation = get_response_transformation(args.response_transformation)

    for test_mode in args.test_mode:
        drug_response_experiment(
            models=models,
            baselines=baselines,
            response_data=response_data,
            response_transformation=response_transformation,
            hpam_optimization_metric=args.optim_metric,
            n_cv_splits=args.n_cv_splits,
            multiprocessing=args.multiprocessing,
            test_mode=test_mode,
            randomization_mode=args.randomization_mode,
            randomization_type=args.randomization_type,
            n_trials_robustness=args.n_trials_robustness,
            cross_study_datasets=cross_study_datasets,
            path_out=args.path_out,
            run_id=args.run_id,
            overwrite=args.overwrite,
            path_data=args.path_data,
            model_checkpoint_dir=args.model_checkpoint_dir,
            hyperparameter_tuning=not args.no_hyperparameter_tuning,
            final_model_on_full_data=args.final_model_on_full_data,
            wandb_project=args.wandb_project,
            custom_splitter=getattr(args, "custom_splitter_path", None),
            custom_split_name=getattr(args, "custom_split_name", None),
            hpo_num_samples=getattr(args, "hpo_num_samples", 16),
            hpo_random_state=getattr(args, "hpo_random_state", 42),
            hpo_resources_per_trial=getattr(args, "hpo_resources_per_trial", None),
        )


def get_datasets(
    dataset_name: str,
    cross_study_datasets: list,
    path_data: str = "data",
    measure: str = "response",
    curve_curator: bool = False,
    cores: int = 1,
    normalize: bool = False,
) -> tuple[DrugResponseDataset, list[DrugResponseDataset] | None]:
    """Load the primary response dataset and optional cross-study datasets.

    Args:
        dataset_name: Built-in or custom dataset name passed to ``load_dataset``.
        cross_study_datasets: Names of additional datasets to load.
        path_data: Root directory for dataset files.
        measure: Response column name.
        curve_curator: Whether to fit CurveCurator for custom datasets.
        cores: Worker count for CurveCurator fitting.
        normalize: Normalize responses during CurveCurator fitting.

    Returns:
        Tuple of the primary dataset and loaded cross-study datasets.
    """
    response_data = load_dataset(
        dataset_name=dataset_name,
        path_data=path_data,
        measure=measure,
        curve_curator=curve_curator,
        cores=cores,
        normalize=normalize,
    )

    cross_study_datasets = [
        load_dataset(dataset_name=dn, path_data=path_data, measure=measure) for dn in cross_study_datasets
    ]
    return response_data, cross_study_datasets
