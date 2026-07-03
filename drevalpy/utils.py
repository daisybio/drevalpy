"""Utility functions for the evaluation pipeline."""

from pathlib import Path

from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .datasets import AVAILABLE_DATASETS
from .datasets.dataset import DrugResponseDataset
from .datasets.loader import load_dataset
from .datasets.splits import validate_split_label
from .datasets.utils import ALLOWED_MEASURES
from .evaluation import AVAILABLE_METRICS
from .experiment import drug_response_experiment, pipeline_function
from .models import MODEL_FACTORY


def check_arguments(args) -> None:
    """
    Check the validity of the arguments for the evaluation pipeline.

    :param args: arguments passed from the command line
    :raises AssertionError: if any of the arguments is invalid
    :raises ValueError: if the number of cross-validation splits or curve_curator_cores is less than 1
    :raises FileNotFoundError: if a custom dataset name was specified and the input file could not be found.
    """
    if not args.models:
        raise AssertionError("At least one model must be specified")
    if not all(model in MODEL_FACTORY for model in args.models):
        raise AssertionError(
            f"Invalid model name. Available models are {list(MODEL_FACTORY.keys())}. If you want to "
            f"use your own model, you need to implement a new model class and add it to the "
            f"MODEL_FACTORY in the models init"
        )
    if not all(test in ["LPO", "LCO", "LDO", "LTO"] for test in args.test_mode):
        raise AssertionError("Invalid test mode. Available test modes are LPO, LCO, LDO, LTO")

    if args.baselines is not None:
        if not all(baseline in MODEL_FACTORY for baseline in args.baselines):
            raise AssertionError(
                f"Invalid baseline name. Available baselines are {list(MODEL_FACTORY.keys())}. If you "
                f"want to use your own baseline, you need to implement a new model class and add it to "
                f"the MODEL_FACTORY in the models init"
            )
    if args.dataset_name not in AVAILABLE_DATASETS:
        if not args.no_refitting:
            expected_custom_input = Path(args.path_data).absolute() / args.dataset_name / f"{args.dataset_name}_raw.csv"
            if not expected_custom_input.is_file():
                raise FileNotFoundError(
                    "You specified the curve_curator option with a custom dataset name which requires raw "
                    f"viability data to be located at {expected_custom_input} but the file does not exist. "
                    "Please check the 'path_data' and 'dataset_name' arguments and ensure the raw viability "
                    "input file is located at <path_data>/<dataset_name>/<dataset_name>_raw.csv."
                )
        else:
            expected_custom_input = Path(args.path_data).absolute() / args.dataset_name / f"{args.dataset_name}.csv"
            if not expected_custom_input.is_file():
                raise FileNotFoundError(
                    "You specified a custom dataset name which requires prefit curve data to be located at "
                    f"{expected_custom_input} but the file does not exist. Please check the 'path_data' and "
                    "'dataset_name' arguments and ensure the prefit curve data is located at input file is "
                    "located at <path_data>/<dataset_name>/<dataset_name>.csv."
                )

    if (not args.no_refitting) and args.curve_curator_cores < 1:
        raise ValueError("Number of cores for CurveCurator must be greater than 0.")

    for dataset in args.cross_study_datasets:
        if dataset not in AVAILABLE_DATASETS:
            raise AssertionError(
                f"Invalid dataset name in cross_study_datasets. Available datasets are "
                f"{list(AVAILABLE_DATASETS.keys())} If you want to use your own dataset, you "
                f"need to implement a new response dataset loader and add it to the "
                f"AVAILABLE_DATASETS in the response_datasets init."
            )

    # if the path to args.path_data does not exist, create the directory
    Path(args.path_data).mkdir(parents=True, exist_ok=True)

    if args.n_cv_splits <= 1 and not getattr(args, "custom_splitter_path", None):
        raise ValueError("Number of cross-validation splits must be greater than 1.")

    custom_splitter_path = getattr(args, "custom_splitter_path", None)
    if custom_splitter_path:
        if not Path(custom_splitter_path).expanduser().is_file():
            raise FileNotFoundError(f"Custom split script not found: {custom_splitter_path}")

    custom_split_name = getattr(args, "custom_split_name", None)
    if custom_split_name is not None:
        validate_split_label(custom_split_name)

    # TODO Allow for custom randomization tests maybe via config file
    if args.randomization_mode[0] != "None":
        if not all(randomization in ["SVCC", "SVRC", "SVCD", "SVRD"] for randomization in args.randomization_mode):
            raise AssertionError(
                "At least one invalid randomization mode. Available randomization modes are SVCC, SVRC, SVCD, SVRD."
            )

    if args.randomization_type not in ["permutation", "invariant"]:
        raise AssertionError("Invalid randomization type. Choose from 'permutation' or 'invariant'")

    if args.n_trials_robustness < 0:
        raise ValueError("Number of trials for robustness test must be greater than or equal to 0")

    if args.measure not in ALLOWED_MEASURES:
        raise ValueError(
            "Only 'LN_IC50', 'EC50', 'IC50', 'pEC50', 'AUC', 'response' or their equivalents including "
            "the '_curvecurator' suffix are allowed drug response measures."
        )

    if args.response_transformation not in ["None", "standard", "minmax", "robust"]:
        raise AssertionError("Invalid response_transformation. Choose from None, standard, minmax, robust")

    if args.optim_metric not in AVAILABLE_METRICS:
        raise AssertionError(
            f"Invalid optim_metric for hyperparameter tuning. Choose from" f" {list(AVAILABLE_METRICS.keys())}"
        )


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

    models = [MODEL_FACTORY[model] for model in args.models]

    if args.baselines is not None:
        baselines = [MODEL_FACTORY[baseline] for baseline in args.baselines]
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
