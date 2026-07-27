"""Utility functions for the evaluation pipeline."""

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .datasets.dataset import DrugResponseDataset
from .datasets.loader import load_dataset
from .experiment import drug_response_experiment, pipeline_function
from .models._model_lookup import get_model_class
from .utils_validation import check_arguments

__all__ = ["check_arguments", "get_datasets", "get_response_transformation", "main"]


def main(args) -> None:
    """
    Main function to run the drug response evaluation pipeline.

    :param args: passed from command line
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
    """
    Load the response data and cross-study datasets.

    :param dataset_name: The name of the dataset to load. Can be one of ('GDSC1', 'GDSC2', 'CCLE', CTRPv1',
        'CTRPv2', 'TOYv1', 'TOYv2')
        to download provided datasets, or any other name to use a custom datasets.
    :param cross_study_datasets: list of cross-study datasets. CurveCurator is not applicable to these. If you wish
        to provide custom cross_study_datasets, you have to invoke curve fitting manually using
        drevalpy.datasets.curvecurator.fit_curves
    :param path_data: The parent path in which custom or downloaded datasets should be located, or in which raw
        viability data is to be found for fitting with CurveCurator (see param curve_curator for details).
        The location of the datasets are resolved by <path_data>/<dataset_name>/<dataset_name>.csv.
    :param measure: The name of the column containing the measure to predict, default = "response".
        If curve_curator is True, this measure is appended with "_curvecurator", e.g. "response_curvecurator" to
        distinguish between measures provided by the original source of a dataset, or the measures fit by
        CurveCurator.
    :param curve_curator: If True, the measure is appended with "_curvecurator".
        If a custom dataset_name was provided, this will invoke the fitting procedure of raw viability data,
        which is expected to exist at <path_data>/<dataset_name>/<dataset_name>_raw.csv. The fitted dataset will
        be stored in the same folder, in a file called <dataset_name>.csv
    :param cores: Number of cores to use for CurveCurator fitting. Only used when curve_curator is True, default = 1
    :param normalize: Whether to normalize the response values to [0, 1] for curvecurator. Default = False.
        Only used for custom datasets when curve_curator is True.
    :returns: response data and, potentially, cross-study datasets
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


@pipeline_function
def get_response_transformation(response_transformation: str | None) -> TransformerMixin | None:
    """
    Get the skelarn response transformation object of choice.

    Users can choose from "None", "standard", "minmax", "robust".

    :param response_transformation: response transformation to apply
    :returns: response transformation object
    :raises ValueError: if the response transformation is not recognized
    """
    if (response_transformation == "None") or (response_transformation is None):
        return None
    if response_transformation == "standard":
        return StandardScaler()
    if response_transformation == "minmax":
        return MinMaxScaler()
    if response_transformation == "robust":
        return RobustScaler()
    raise ValueError(
        f"Unknown response transformation {response_transformation}. Choose from 'None', "
        f"'standard', 'minmax', 'robust'"
    )
