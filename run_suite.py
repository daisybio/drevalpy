import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from models import MODEL_FACTORY
from response_datasets import RESPONSE_DATASET_FACTORY
from suite.experiment import drug_response_experiment


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run the drug response prediction model test suite"
    )
    parser.add_argument(
        "--run_id", type=str, default="my_run", help="identifier to save the results"
    )
    parser.add_argument(
        "--models", nargs="+", help="model to evalaute or list of models to compare"
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
        "For each experiment one drug view is randomized while the others are held constant."
        ,
    )
    parser.add_argument(
        "--randomization_type",
        type=str,
        default="permutation",
        help="""type of randomization to use. Choose from "gaussian", "zeroing", "permutation". Default is "permutation"
            "gaussian": replace the features with random values sampled from a gaussian distribution with the same mean and standard deviation
            "zeroing": replace the features with zeros
            "permutation": permute the features over the instances, keeping the distribution of the features the same but dissolving the relationship to the target"""
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="GDSC1",
        help="Name of the drug response dataset"
    )

    parser.add_argument(
        "--path_out", type=str, default="results/", help="Path to the output directory"
    )

    parser.add_argument(
        "--curve_curator",
        action='store_true',
        default=False,
        help="Whether to run " "CurveCurator " "to sort out " "non-reactive " "curves",
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        default=False,
        help="Overwrite existing results with the same path out and run_id? ",
    )
    parser.add_argument(
        "--optim_metric",
        type=str,
        default="RMSE",
        help="Metric to optimize for (RMSE, AUC, ACC, F1, MCC, R2, etc.)"
    )
    parser.add_argument(
        "--response_transformation",
        type=str,
        default="None",
        help="Transformation to apply to the response variable possible values: standard, minmax, robust")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    assert args.models, "At least one model must be specified"
    assert all(
        [model in MODEL_FACTORY for model in args.models]
    ), f"Invalid model name. Available models are {list(MODEL_FACTORY.keys())}. If you want to use your own model, you need to implement a new model class and add it to the MODEL_FACTORY in the models init"
    assert all(
        [test in ["LPO", "LCO", "LDO"] for test in args.test_mode]
    ), "Invalid test mode. Available test modes are LPO, LCO, LDO"
    models = [
        MODEL_FACTORY[model](model_name=model, target="IC50") for model in args.models
    ]
    assert args.dataset_name in RESPONSE_DATASET_FACTORY, f"Invalid dataset name. Available datasets are {list(RESPONSE_DATASET_FACTORY.keys())} If you want to use your own dataset, you need to implement a new response dataset class and add it to the RESPONSE_DATASET_FACTORY in the response_datasets init"
    response_data = RESPONSE_DATASET_FACTORY[args.dataset_name]()

    if args.randomization_mode[0] != "None":
        assert all(
            [randomization in ["SVCC", "SVRC", "SVSC", "SVRD"] for randomization in args.randomization_mode]
        ), "At least one invalid randomization mode. Available randomization modes are SVCC, SVRC, SVSC, SVRD"
    else:
        args.randomization_mode = None
    if args.curve_curator:
        raise NotImplementedError("CurveCurator not implemented")
    if args.response_transformation == "None":
        response_transformation = None
    elif(args.response_transformation == "standard"):
        response_transformation = StandardScaler()
    elif(args.response_transformation == "minmax"):
        response_transformation = MinMaxScaler()
    elif(args.response_transformation == "robust"):
        response_transformation = RobustScaler()
    else:
        raise ValueError(f"Invalid response_transformation: {args.response_transformation}. Choose robust, minmax or standard.")

    # TODO Allow for custom randomization tests maybe via config file 



    # TODO metric for optimization needs to be considered
    for test_mode in args.test_mode:
        drug_response_experiment(
            models=models,
            response_data=response_data,
            response_transformation=response_transformation,
            multiprocessing=True,
            test_mode=test_mode,
            randomization_mode=args.randomization_mode,
            randomization_type=args.randomization_type,
            path_out=args.path_out,
            run_id=args.run_id,
            overwrite=args.overwrite,

        )

    # TODO now do evaluation, visualization, etc.

