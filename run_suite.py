import argparse

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
        default="LPO",
        help="Which tests to run (LPO=Leave-random-Pairs-Out, "
        "LCO=Leave-Cell-line-Out, LDO=Leave-Drug-Out). Can be a list of test runs e.g. 'LPO LCO LDO' to run all tests. Default is LPO",
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


    if args.curve_curator:
        raise NotImplementedError("CurveCurator not implemented")

    # TODO randomization_test_views need to be specified. maybe via config file 
    # TODO metric for optimization needs to be considered
    for test_mode in args.test_mode:
        drug_response_experiment(
            models,
            response_data,
            multiprocessing=True,
            test_mode=test_mode,
            randomization_test_views={"randomize_gene_expression": ["gene_expression"], "randomize_genomics": ["mutation", "copy_number_var"]},
            path_out=args.path_out,
            run_id=args.run_id,
            overwrite=args.overwrite,
        )

    # TODO now do evaluation, visualization, etc.

