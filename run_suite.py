# Main file for running the whole suite of tests

import os
import argparse
import pandas as pd
import numpy as np

from suite.data_wrapper import DrugResponseDataset, FeatureDataset


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run the drug response prediction model test suite"
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        default="LPO",
        help="Which tests to run (LPO=Leave-random-Pairs-Out, "
        "LCO=Leave-Cell-line-Out, LDO=Leave-Drug-Out)",
    )
    parser.add_argument(
        "--path_dr",
        type=str,
        default="data/drug_response/",
        help="Path to the drug response dataset "
        "/ directory containing all "
        "relevant files",
    )
    parser.add_argument(
        "--name_dr",
        type=str,
        default="CCLE",
        help="Name of the drug response " "dataset",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="IC50",
        help="Target variable to predict (AUC, IC50, EC50, "
        "classification into sensitive/resistant)",
    )

    parser.add_argument(
        '--column_response',
        type=str,
        default='IC50',
        help='Name of the column containing the response values'
    )

    parser.add_argument(
        '--column_drug',
        type=str,
        default='Drug',
        help='Name of the column containing the drug IDs'
    )

    parser.add_argument(
        '--column_cell_line',
        type=str,
        default='Cell_line',
        help='Name of the column containing the cell line IDs'
    )

    parser.add_argument(
        "--path_df",
        type=str,
        default="data/drug_features.csv",
        help="Path to csv file containing all drug features. 2 columns: Feature,Path with feature name and path to "
             "file containing that feature"
    )

    parser.add_argument(
        "--path_cf",
        type=str,
        default="data/cell_line_features.csv",
        help="Path to csv file containing all cell line features. 2 columns: Feature,Path with feature name and path to"
             "file containing that feature"
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
        "--custom_models",
        action='store_true',
        default=False,
        help="Whether to "
        "train a custom"
        " model "
        "provided by "
        "the user via "
        "the wrapper",
    )

    parser.add_argument(
        "--optim_metric",
        type=str,
        default="RMSE",
        help="Metric to optimize for (RMSE, AUC, ACC, F1, MCC, R2, etc.)"
    )

    return parser


def compare_models(
    models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
):
    """
    Compare the models in the list models on the given datasets.
    """
    for model in models:
        # Train the model
        # Test the model
        # Save the results
        pass


def robustness_tests(
    models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
):
    """
    Run the randomization tests.
    """
    train_with_different_seed(
        models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
    )
    train_with_permuted_input(
        models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
    )


def train_with_different_seed(
    models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
):
    """
    Train the models with different random seeds.
    """
    pass


def train_with_permuted_input(
    models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
):
    """
    Train the models with input in permuted order.
    """
    pass


def randomization_tests(
    models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
):
    """
    Run the randomization tests.
    """
    train_with_shuffled_input(
        models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
    )
    train_with_zero_inputs(
        models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
    )


def train_with_shuffled_input(
    models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
):
    """
    Train the models with shuffled input.
    """
    pass


def train_with_zero_inputs(
    models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
):
    """
    Train the models with zero vectors as inputs.
    """
    pass


def visualize_results(
    models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
):
    """
    Visualize the results.
    """
    pass


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(args)

    # Get which models should be run
    if args.target == "classification":
        admissible_metrics = ["AUC", "ACC", "F1", "MCC"]
        if args.optim_metric not in admissible_metrics:
            raise ValueError(f"Metric not valid for classification! Please choose one of {admissible_metrics}")
        models = ["Baselines", "DeepCDR", "MOLI", "Super.FELT"]
        print(f"Running classification tests on the models {models}")
    else:
        admissible_metrics = ["RMSE", "R2", "MSE", "MAE"]
        if args.optim_metric not in admissible_metrics:
            raise ValueError(f"Metric not valid for regression! Please choose one of {admissible_metrics}")
        if args.target == "AUC":
            models = ["Baselines", "DrugCell"]
        elif args.target == "IC50":
            models = ["Baselines", "BiGDRP", "DeepCDR", "PaccMan", "SRMF"]
        elif args.target == "EC50":
            models = ["Baselines"]
        else:
            raise ValueError("Target variable not recognized")
        print(
            f"Running regression tests on the models {models} with target variable {args.target}"
        )

    # Make DRP dataset from path and split it according to test mode
    response_df = pd.read_csv(args.path_dr)
    # drop everything except for the relevant 3 columns and reshape: row names are cell lines and column names are drugs
    response_df = response_df[[args.column_cell_line, args.column_drug, args.column_response]]
    # get duplicates of cell line/drug combinations
    duplicates = response_df.duplicated(subset=[args.column_cell_line, args.column_drug])
    # print duplicates
    print(f'The following cell line/drug combinations are duplicated: {response_df[duplicates]}')
    print('Dropping duplicates...')
    # drop duplicates
    response_df = response_df.drop_duplicates(subset=[args.column_cell_line, args.column_drug])
    # reshape long to wide
    response_df = response_df.pivot(index=args.column_cell_line, columns=args.column_drug, values=args.column_response)
    # make response dataset
    drp_dataset = DrugResponseDataset(
        response=response_df.values,
        cell_line_ids=response_df.index.values,
        drug_ids=response_df.columns.values
    )

    # Make drug feature dataset from path
    drug_feature_list = pd.read_csv(args.path_df)
    feature_names = drug_feature_list['Feature'].values
    feature_paths = drug_feature_list['Path'].values
    features = {}
    for feature_name, feature_path in zip(feature_names, feature_paths):
        if not os.path.exists(feature_path):
            raise ValueError(f'Path to {feature_name} does not exist!')
        # make dictionary of drug IDs (column names) and feature vectors (column values)
        feature_dict = {}
        drug_feature_df = pd.read_csv(feature_path)
        for drug_id in drug_feature_df.columns.values:
            if drug_id == 'Unnamed: 0':
                continue
            feature_dict[drug_id] = drug_feature_df[drug_id].values
        features[feature_name] = feature_dict
    drug_feature_dataset = FeatureDataset(
        features=features
    )

    # Make cell line feature dataset from path
    cell_line_feature_list = pd.read_csv(args.path_cf)
    feature_names = cell_line_feature_list['Feature'].values
    feature_paths = cell_line_feature_list['Path'].values
    features = {}
    for feature_name, feature_path in zip(feature_names, feature_paths):
        if not os.path.exists(feature_path):
            raise ValueError(f'Path to {feature_name} does not exist!')
        # make dictionary of cell line IDs (column names) and feature vectors (column values)
        feature_dict = {}
        cell_line_feature_df = pd.read_csv(feature_path)

        for row in cell_line_feature_df.values:
            cell_line_id = row[0]
            feature_dict[cell_line_id] = row[1:]
        features[feature_name] = feature_dict

    cell_line_feature_dataset = FeatureDataset(
        features=features
    )

    if args.curve_curator:
        # Run CurveCurator
        pass

    if args.custom_models:
        # Run custom models
        pass

    # Compare the models
    compare_models(models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset)

    # Run the robustness tests
    robustness_tests(
        models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
    )

    # Run the randomization tests
    randomization_tests(
        models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
    )

    # Run the visualization
    visualize_results(
        models, drp_dataset, drug_feature_dataset, cell_line_feature_dataset
    )