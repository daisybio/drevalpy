"""CLI argument validation helpers for the evaluation pipeline."""

from pathlib import Path

from .datasets import AVAILABLE_DATASETS
from .datasets.splits import validate_split_label
from .datasets.utils import ALLOWED_MEASURES
from .evaluation import AVAILABLE_METRICS
from .models._model_lookup import known_model_names

_VALID_TEST_MODES = frozenset({"LPO", "LCO", "LDO", "LTO"})
_VALID_RANDOMIZATION_MODES = frozenset({"SVCC", "SVRC", "SVCD", "SVRD"})
_VALID_RESPONSE_TRANSFORMS = frozenset({"None", "standard", "minmax", "robust"})


def validate_models(args) -> None:
    """Validate primary and baseline model names."""
    available_models = known_model_names(include_external=True)
    if not args.models:
        raise AssertionError("At least one model must be specified")
    if not all(model in available_models for model in args.models):
        raise AssertionError(
            f"Invalid model name. Available models are {available_models}. If you want to "
            f"use your own model, register a zoo YAML under the external zoo path (or "
            f"package zoo) and resolve it by name; custom recipe strings are available "
            f"programmatically via construct_model(name, spec) / ModelConfig.from_spec."
        )
    if args.baselines is None:
        return
    if not all(baseline in available_models for baseline in args.baselines):
        raise AssertionError(
            f"Invalid baseline name. Available baselines are {available_models}. If you "
            f"want to use your own baseline, register components and a zoo preset, then "
            f"resolve it with construct_model / ModelConfig.from_spec."
        )


def validate_test_modes(args) -> None:
    """Validate test mode strings."""
    if not all(test in _VALID_TEST_MODES for test in args.test_mode):
        raise AssertionError("Invalid test mode. Available test modes are LPO, LCO, LDO, LTO")


def _expected_custom_dataset_path(args) -> Path:
    base = Path(args.path_data).absolute() / args.dataset_name
    if not args.no_refitting:
        return base / f"{args.dataset_name}_raw.csv"
    return base / f"{args.dataset_name}.csv"


def _custom_dataset_error_message(args, expected: Path) -> str:
    if not args.no_refitting:
        return (
            "You specified the curve_curator option with a custom dataset name which requires raw "
            f"viability data to be located at {expected} but the file does not exist. "
            "Please check the 'path_data' and 'dataset_name' arguments and ensure the raw viability "
            "input file is located at <path_data>/<dataset_name>/<dataset_name>_raw.csv."
        )
    return (
        "You specified a custom dataset name which requires prefit curve data to be located at "
        f"{expected} but the file does not exist. Please check the 'path_data' and "
        "'dataset_name' arguments and ensure the prefit curve data is located at input file is "
        "located at <path_data>/<dataset_name>/<dataset_name>.csv."
    )


def validate_dataset_name_and_paths(args) -> None:
    """Validate built-in or custom dataset paths."""
    if args.dataset_name in AVAILABLE_DATASETS:
        return
    expected = _expected_custom_dataset_path(args)
    if not expected.is_file():
        raise FileNotFoundError(_custom_dataset_error_message(args, expected))


def validate_curve_curator_cores(args) -> None:
    """Validate CurveCurator core count when refitting is enabled."""
    if (not args.no_refitting) and args.curve_curator_cores < 1:
        raise ValueError("Number of cores for CurveCurator must be greater than 0.")


def validate_cross_study_dataset_names(args) -> None:
    """Validate cross-study dataset identifiers."""
    for dataset in args.cross_study_datasets:
        if dataset not in AVAILABLE_DATASETS:
            raise AssertionError(
                f"Invalid dataset name in cross_study_datasets. Available datasets are "
                f"{list(AVAILABLE_DATASETS.keys())} If you want to use your own dataset, you "
                f"need to implement a new response dataset loader and add it to the "
                f"AVAILABLE_DATASETS in the response_datasets init."
            )


def validate_cv_split_settings(args) -> None:
    """Validate CV split count and custom split script paths."""
    if args.n_cv_splits <= 1 and not getattr(args, "custom_splitter_path", None):
        raise ValueError("Number of cross-validation splits must be greater than 1.")

    custom_splitter_path = getattr(args, "custom_splitter_path", None)
    if custom_splitter_path and not Path(custom_splitter_path).expanduser().is_file():
        raise FileNotFoundError(f"Custom split script not found: {custom_splitter_path}")

    custom_split_name = getattr(args, "custom_split_name", None)
    if custom_split_name is not None:
        validate_split_label(custom_split_name)


def validate_randomization_settings(args) -> None:
    """Validate randomization mode and type."""
    if args.randomization_mode[0] == "None":
        return
    if not all(mode in _VALID_RANDOMIZATION_MODES for mode in args.randomization_mode):
        raise AssertionError(
            "At least one invalid randomization mode. Available randomization modes are SVCC, SVRC, SVCD, SVRD."
        )
    if args.randomization_type not in ["permutation", "invariant"]:
        raise AssertionError("Invalid randomization type. Choose from 'permutation' or 'invariant'")


def validate_robustness_and_hpo_settings(args) -> None:
    """Validate robustness trial count and HPO sample count."""
    if args.n_trials_robustness < 0:
        raise ValueError("Number of trials for robustness test must be greater than or equal to 0")

    hpo_num_samples = getattr(args, "hpo_num_samples", 16)
    if hpo_num_samples < 0:
        raise ValueError("hpo_num_samples must be greater than or equal to 0")


def validate_measure_and_metrics(args) -> None:
    """Validate response measure, transformation, and optimization metric."""
    if args.measure not in ALLOWED_MEASURES:
        raise ValueError(
            "Only 'LN_IC50', 'EC50', 'IC50', 'pEC50', 'AUC', 'response' or their equivalents including "
            "the '_curvecurator' suffix are allowed drug response measures."
        )
    if args.response_transformation not in _VALID_RESPONSE_TRANSFORMS:
        raise AssertionError("Invalid response_transformation. Choose from None, standard, minmax, robust")
    if args.optim_metric not in AVAILABLE_METRICS:
        raise AssertionError(
            f"Invalid optim_metric for hyperparameter tuning. Choose from {list(AVAILABLE_METRICS.keys())}"
        )


def check_arguments(args) -> None:
    """
    Check the validity of the arguments for the evaluation pipeline.

    :param args: arguments passed from the command line
    :raises AssertionError: if any of the arguments is invalid
    :raises ValueError: if the number of cross-validation splits or curve_curator_cores is less than 1
    :raises FileNotFoundError: if a custom dataset name was specified and the input file could not be found.
    """
    validate_models(args)
    validate_test_modes(args)
    validate_dataset_name_and_paths(args)
    validate_curve_curator_cores(args)
    validate_cross_study_dataset_names(args)
    Path(args.path_data).mkdir(parents=True, exist_ok=True)
    validate_cv_split_settings(args)
    validate_randomization_settings(args)
    validate_robustness_and_hpo_settings(args)
    validate_measure_and_metrics(args)
