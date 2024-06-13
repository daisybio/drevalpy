import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from drevalpy.models import MODEL_FACTORY
from drevalpy.datasets import RESPONSE_DATASET_FACTORY
from drevalpy.experiment import drug_response_experiment
from drevalpy.evaluation import AVAILABLE_METRICS


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run the drug response prediction model test suite."
    )
    parser.add_argument(
        "--run_id", type=str, default="my_run", help="identifier to save the results"
    )
    parser.add_argument(
        "--models", nargs="+", help="model to evaluate or list of models to compare"
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        help="baseline or list of baselines. The baselines are also hpam-tuned and compared to the models, but no randomization or robustness tests are run.",
    )
    parser.add_argument(
        "--test_mode",
        nargs="+",
        default=["LPO"],
        help="Which tests to run (LPO=Leave-random-Pairs-Out, "
        "LCO=Leave-Cell-line-Out, LDO=Leave-Drug-Out). Can be a list of test runs e.g. 'LPO LCO LDO' to run all tests. Default is LPO",
    )
    parser.add_argument(
        "--randomization_mode",
        nargs="+",
        default=["None"],
        help="Which randomization tests to run, additionally to the normal run. Default is None which means no randomization tests are run."
        "Modes: SVCC, SVRC, SVCD, SVRD"
        "Can be a list of randomization tests e.g. 'SCVC SCVD' to run two tests. Default is None"
        "SVCC: Single View Constant for Cell Lines: in this mode, one experiment is done for every cell line view the model uses (e.g. gene expression, mutation, ..)."
        "For each experiment one cell line view is held constant while the others are randomized. "
        "SVRC Single View Random for Cell Lines: in this mode, one experiment is done for every cell line view the model uses (e.g. gene expression, mutation, ..)."
        "For each experiment one cell line view is randomized while the others are held constant."
        "SVCD: Single View Constant for Drugs: in this mode, one experiment is done for every drug view the model uses (e.g. fingerprints, target_information, ..)."
        "For each experiment one drug view is held constant while the others are randomized."
        "SVRD: Single View Random for Drugs: in this mode, one experiment is done for every drug view the model uses (e.g. gene expression, target_information, ..)."
        "For each experiment one drug view is randomized while the others are held constant.",
    )
    parser.add_argument(
        "--randomization_type",
        type=str,
        default="permutation",
        help="""type of randomization to use. Choose from "gaussian", "zeroing", "permutation". Default is "permutation"
            "gaussian": replace the features with random values sampled from a gaussian distribution with the same mean and standard deviation
            "zeroing": replace the features with zeros
            "permutation": permute the features over the instances, keeping the distribution of the features the same but dissolving the relationship to the target""",
    )
    parser.add_argument(
        "--n_trials_robustness",
        type=int,
        default=0,
        help="Number of trials to run for the robustness test. Default is 0, which means no robustness test is run. The robustness test is a test where the model is trained with varying seeds. This is done multiple times to see how stable the model is.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="GDSC1",
        help="Name of the drug response dataset",
    )

    parser.add_argument(
        "--cross_study_datasets",
        nargs="+",
        default=[],
        help="List of datasets to use to evaluate predictions acresso studies. Default is empty list which means no cross-study datasets are used.",
    )

    parser.add_argument(
        "--path_out", type=str, default="results/", help="Path to the output directory"
    )

    parser.add_argument(
        "--curve_curator",
        action="store_true",
        default=False,
        help="Whether to run " "CurveCurator " "to sort out " "non-reactive " "curves",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing results with the same path out and run_id? ",
    )
    parser.add_argument(
        "--optim_metric",
        type=str,
        default="RMSE",
        help=f"Metric for hyperparameter tuning choose from {list(AVAILABLE_METRICS.keys())} Default is RMSE.",
    )
    parser.add_argument(
        "--n_cv_splits",
        type=int,
        default=5,
        help="Number of cross-validation splits to use for the evaluation",
    )

    parser.add_argument(
        "--response_transformation",
        type=str,
        default="None",
        help="Transformation to apply to the response variable possible values: standard, minmax, robust",
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        default=False,
        help="Whether to use multiprocessing for the evaluation. Default is False",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    assert args.models, "At least one model must be specified"
    assert all(
        [model in MODEL_FACTORY for model in args.models]
    ), f"Invalid model name. Available models are {list(MODEL_FACTORY.keys())}. If you want to use your own model, you need to implement a new model class and add it to the MODEL_FACTORY in the models init"
    assert all(
        [baseline in MODEL_FACTORY for baseline in args.baselines]
    ), f"Invalid baseline name. Available baselines are {list(MODEL_FACTORY.keys())}. If you want to use your own baseline, you need to implement a new model class and add it to the MODEL_FACTORY in the models init"
    assert all(
        [test in ["LPO", "LCO", "LDO"] for test in args.test_mode]
    ), "Invalid test mode. Available test modes are LPO, LCO, LDO"
    models = [MODEL_FACTORY[model] for model in args.models]
    if args.baselines:
        baselines = [MODEL_FACTORY[baseline] for baseline in args.baselines]
    else:
        baselines = []
    assert (
        args.dataset_name in RESPONSE_DATASET_FACTORY
    ), f"Invalid dataset name. Available datasets are {list(RESPONSE_DATASET_FACTORY.keys())} If you want to use your own dataset, you need to implement a new response dataset class and add it to the RESPONSE_DATASET_FACTORY in the response_datasets init"
    for dataset in args.cross_study_datasets:
        assert (
            dataset in RESPONSE_DATASET_FACTORY
        ), f"Invalid dataset name in cross_study_datasets. Available datasets are {list(RESPONSE_DATASET_FACTORY.keys())} If you want to use your own dataset, you need to implement a new response dataset class and add it to the RESPONSE_DATASET_FACTORY in the response_datasets init"

    response_data = RESPONSE_DATASET_FACTORY[args.dataset_name]()
    cross_study_datasets = [
        RESPONSE_DATASET_FACTORY[dataset]() for dataset in args.cross_study_datasets
    ]
    assert (
        args.n_cv_splits > 1
    ), "Number of cross-validation splits must be greater than 1"

    # TODO Allow for custom randomization tests maybe via config file
    if args.randomization_mode[0] != "None":
        assert all(
            [
                randomization in ["SVCC", "SVRC", "SVSC", "SVRD"]
                for randomization in args.randomization_mode
            ]
        ), "At least one invalid randomization mode. Available randomization modes are SVCC, SVRC, SVSC, SVRD"
    else:
        args.randomization_mode = None
    if args.curve_curator:
        raise NotImplementedError("CurveCurator not implemented")
    if args.response_transformation == "None":
        response_transformation = None
    elif args.response_transformation == "standard":
        response_transformation = StandardScaler()
    elif args.response_transformation == "minmax":
        response_transformation = MinMaxScaler()
    elif args.response_transformation == "robust":
        response_transformation = RobustScaler()
    else:
        # TODO add log, sinh etc
        raise ValueError(
            f"Invalid response_transformation: {args.response_transformation}. Choose robust, minmax or standard."
        )

    assert (
        args.optim_metric in AVAILABLE_METRICS
    ), f"Invalid optim_metric for hyperparameter tuning. Choose from {list(AVAILABLE_METRICS.keys())}"

    for test_mode in args.test_mode:
        drug_response_experiment(
            models=models,
            baselines=baselines,
            response_data=response_data,
            response_transformation=response_transformation,
            metric=args.optim_metric,
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
        )

    # TODO now do evaluation, visualization, etc.
